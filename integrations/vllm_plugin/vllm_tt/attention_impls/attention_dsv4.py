# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""
DeepSeek-V4 (DSV4) attention backend for TT devices.

DSV4 attention is *not* one more MLA variant. Relative to the DeepSeek-V3-style
MLA that ``attention_mla.py`` handles, a DSV4 attention layer is typed by its
``compress_ratio``:

    compress_ratio <= 1   ("SWA-only")   sliding window only, normalized alone
    compress_ratio == 4   ("C4A"/CSA)    window + lightning-indexer top-k branch
    compress_ratio == 128 ("C128A"/HCA)  window + contiguous compressed prefix

Every layer additionally folds a per-head *attention sink* logit into the
softmax denominator. The C4A / C128A layers run a second (compressed) branch in
parallel with the window branch and merge the two with an online-softmax stitch.

This module implements the **SWA-only** slice end-to-end (the first milestone):
windowed MLA attention with an attention sink, a single normalized branch, no
compressed branch and no merge. It reuses the existing TT MLA kernels:

    prefill : tt.flash_mla_prefill      + a banded (windowed-causal) attn_mask
                                          + an LSE-based sink fold in this impl
    decode  : tt.paged_flash_mla_decode + native sliding_window + attention_sink

See ``DSV4_TT_Attention_Design.md`` and the "what's next" TODOs at the bottom
for the compressed / indexer / merge branches (out of scope for this milestone).

Design note — why the upstream layer is *not* subclassed here. vLLM's
``DeepseekV4MultiHeadLatentAttentionWrapper`` is hard-bound to CUDA: its
``__init__`` asserts ``get_device_capability() is not None`` and it uses
``torch.cuda.Event`` / fp8 einsum / the FlashMLA sparse kernels. It cannot be
constructed on the ``tt`` platform. The numerically-validated deliverable is
therefore this impl class, exercised in isolation (mirroring
``tests/.../oot_backends/test_mla_attention_impl.py``), rather than a full-model
forward. Full-model wiring (platform gating, multi-group KV, per-branch
metadata, a ``PluggableLayer.register(...)`` TT wrapper) is scaffolded but gated;
see ``platform.py`` / ``model_runner.py`` and the design doc.
"""
from typing import TYPE_CHECKING, Optional

import torch
from vllm.v1.attention.backend import AttentionBackend, MLAAttentionImpl

from ..logger import tt_init_logger
from .attention import TTAttentionMetadataBuilder, TTMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

logger = tt_init_logger(__name__)

_NEG_INF = float("-inf")

# The ttnn flash_mla_prefill kernel requires head_dim_v < qk head dim. DSV4's
# direct MLA uses V = the full latent (head_dim_v == qk), so we zero-pad the
# qk head dim by one tile: V = key[..., :head_dim_v] stays the real latent and
# the padded dims contribute 0 to the scores. Validated on tt (PCC ~0.9998).
_MLA_V_PAD = 32


def _is_bound_cache(kv_cache) -> bool:
    """True when a real paged KV cache is bound (vs the empty [] placeholder)."""
    return isinstance(kv_cache, torch.Tensor) and kv_cache.numel() > 0


# --------------------------------------------------------------------------- #
# Decoupled RoPE helpers (DSV4 uses GPT-J style rope on the rope_head_dim slice)
# --------------------------------------------------------------------------- #
def _dsv4_cos_sin(cos_sin_cache: torch.Tensor, positions: torch.Tensor, rope_dim: int):
    """cos/sin for ``positions`` from a vLLM ``cos_sin_cache`` ([max_pos,
    rope_dim] = concat(cos[rope_dim/2], sin[rope_dim/2])). Returns cos, sin of
    shape ``[T, rope_dim/2]``.

    NOTE: this matches the vLLM cos_sin_cache convention; the exact GPT-J
    interleave should be cross-checked on GPU against DSV4's fused
    ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` op (not runnable here).
    """
    half = rope_dim // 2
    cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    return cache[:, :half], cache[:, half : 2 * half]


def _dsv4_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sign=1.0):
    """GPT-J rope over ``x``'s last dim (adjacent pairs). ``cos``/``sin`` broadcast
    against ``x[..., 0::2]``. ``sign=-1`` inverts (used in the o-proj)."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    c, s = cos.float(), sin.float() * sign
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2).to(x.dtype)


def _dsv4_rope_rope_dims(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope: int,
    headed: bool,
    sign=1.0,
):
    """Rotate only ``x[..., nope:]`` (the rope slice); pass the nope part through.
    ``headed`` broadcasts cos/sin over a heads axis (x is ``[T, N, D]``)."""
    if headed:
        cos, sin = cos[:, None, :], sin[:, None, :]
    roped = _dsv4_apply_rope(x[..., nope:], cos, sin, sign=sign)
    return torch.cat([x[..., :nope], roped], dim=-1)


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class TTDeepseekV4AttentionBackend(AttentionBackend):
    """vLLM attention backend for DeepSeek-V4 MLA attention on TT.

    Generically named because it will host all three DSV4 attention styles
    (SWA-only, C4A, C128A). Only the **SWA-only** slice is implemented today; the
    compressed / indexer branches will reuse this same backend (their compressed
    KV cache is a second cache group — see ``TTDeepseekV4AttentionBackendImpl``).

    The (SWA) cache stores one concatenated latent KV tensor per token, exactly
    like the V3 MLA cache: ``num_kv_heads == 1`` and
    ``head_size = kv_lora_rank + qk_rope_head_dim``. The physical tensor shape is
    identical to ``TTMLAAttentionBackend``'s, so the shape helper is shared.
    """

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4"

    @staticmethod
    def get_impl_cls() -> type["TTDeepseekV4AttentionBackendImpl"]:
        return TTDeepseekV4AttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        return TTAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1, "num_kv_heads must be 1 for DSV4 MLA"
        return (num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_page_size(vllm_config: "VllmConfig") -> int:
        # DeepSeek-V4 forces the SWA cache block_size to 64 (sparse_swa.py:74).
        return 64

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TT DSV4 backend.")


# --------------------------------------------------------------------------- #
# Impl
# --------------------------------------------------------------------------- #
class TTDeepseekV4AttentionBackendImpl(MLAAttentionImpl):
    """DeepSeek-V4 MLA attention impl for TT (currently the SWA-only slice).

    Mirrors ``TTMLAAttentionBackendImpl`` (same latent construction, Q-absorption
    and paged-cache writes) but:
      * stores and applies ``sliding_window``;
      * folds a per-head ``attention_sink`` into the softmax denominator;
      * dispatches by ``compress_ratio`` (only ``<= 1`` is implemented here).

    The attention sink and absorbed weights are read off the ``layer`` object at
    call time (``layer.attn_sink`` / ``layer.W_UK_T`` / ``layer.W_UV``), matching
    how the MLA impl reads ``layer.W_UK_T`` / ``layer.W_UV`` — they are plain
    tensor attributes that ``model.to('xla')`` does not move, so the impl does an
    explicit ``.to(device=...)`` on each.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        # MLA-specific arguments
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: int = 0,
        qk_nope_head_dim: int = 0,
        qk_rope_head_dim: int = 0,
        qk_head_dim: int = 0,
        v_head_dim: int = 0,
        kv_b_proj=None,
        indexer: Optional[object] = None,
        q_pad_num_heads: Optional[int] = None,
        # DSV4-specific
        compress_ratio: int = 1,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        # Unlike the V3 MLA impl (which accepts but ignores sliding_window), DSV4
        # SWA *is* the sliding window, so we store it.
        self.sliding_window = sliding_window
        self.compress_ratio = compress_ratio
        self.indexer = indexer

        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes are not supported for DSV4 on TT.")
        if kv_cache_dtype != "auto":
            # DSV4 upstream uses an fp8_ds_mla latent cache; the first TT
            # milestone uses a bf16 latent cache (fp8 is a later optimization).
            raise NotImplementedError(
                f"Quantized DSV4 KV cache ({kv_cache_dtype}) is not yet supported "
                "on TT; use a bf16 latent cache for now."
            )
        if compress_ratio is not None and compress_ratio > 1:
            raise NotImplementedError(
                f"DSV4 compress_ratio={compress_ratio} (C4A/C128A: compressed "
                "branch + two-branch merge) is not implemented yet on TT. Only "
                "SWA-only layers (compress_ratio <= 1) are supported. See the "
                "'what's next' TODOs in attention_dsv4.py."
            )

    # ------------------------------------------------------------------ #
    # Abstract stubs — never called; the layer routes through forward().
    # ------------------------------------------------------------------ #
    def forward_mha(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "TTDeepseekV4AttentionBackendImpl.forward_mha should never be "
            "called; route through forward() directly."
        )

    def forward_mqa(self, *args, **kwargs):
        raise RuntimeError(
            "TTDeepseekV4AttentionBackendImpl.forward_mqa should never be "
            "called; route through forward() directly."
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_is_prefill(
        q_nope: torch.Tensor, attn_metadata: Optional[TTMetadata]
    ) -> bool:
        if attn_metadata is None or attn_metadata.cache_position is None:
            return True  # profiling runs are treated as prefill
        users = attn_metadata.cache_position.shape[0]
        assert users > 0, "Invalid number of users"
        return (q_nope.shape[0] // users) > 1

    def _get_attn_sink(
        self, layer, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Read the per-head sink logits off ``layer`` (``None`` if absent).

        The sink is a ``[num_heads]`` (or head-padded) tensor. We slice to
        ``num_heads`` and keep it in the requested dtype. Upstream initialises it
        to ``-inf`` (no effect) and weight-loading fills the real values.
        """
        sink = getattr(layer, "attn_sink", None)
        if sink is None:
            return None
        sink = sink.to(device=device)
        if sink.shape[0] > self.num_heads:
            sink = sink[: self.num_heads]
        return sink.to(dtype)

    def _windowed_causal_mask(
        self,
        users: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Additive ``[users, 1, S, S]`` mask: ``0`` where key ``j`` is in the
        causal sliding window ``[i - W + 1, i]`` of query ``i``, ``-inf`` else.

        When ``sliding_window`` is ``None`` this degrades to a plain causal mask.
        Shared by every user (prefill sequences all start at position 0), so we
        build ``[1, 1, S, S]`` and broadcast.
        """
        i = torch.arange(seq_len, device=device)[:, None]
        j = torch.arange(seq_len, device=device)[None, :]
        keep = j <= i
        if self.sliding_window is not None and self.sliding_window > 0:
            keep = keep & (j > i - self.sliding_window)
        mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
        mask = mask.masked_fill(~keep, _NEG_INF)
        return mask.view(1, 1, seq_len, seq_len).expand(users, 1, seq_len, seq_len)

    # ------------------------------------------------------------------ #
    # Unified forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        q: tuple[torch.Tensor, torch.Tensor],
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        layer: "MLAAttention",
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """DeepSeek-V4 SWA-only MLA attention on TT (prefill and paged decode).

        Shapes (from the layer after splitting ``q``):
            q_nope:      [tokens, num_heads, qk_nope_head_dim]
            q_pe:        [tokens, num_heads, qk_rope_head_dim]
            kv_c_normed: [tokens, kv_lora_rank]
            k_pe:        [tokens, 1, qk_rope_head_dim]
            kv_cache:    [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
            output:      [tokens, num_heads * v_head_dim]   (write target)
        """
        assert output is not None, "forward requires an output tensor."
        q_nope, q_pe = q

        is_prefill = self._infer_is_prefill(q_nope, attn_metadata)
        users = (
            attn_metadata.cache_position.shape[0]
            if attn_metadata is not None and attn_metadata.cache_position is not None
            else 1
        )
        total_tokens = q_nope.shape[0]
        assert (
            total_tokens % users == 0
        ), f"total_tokens ({total_tokens}) not divisible by users ({users})."
        S = total_tokens // users
        N = self.num_heads
        L = self.kv_lora_rank
        R = self.qk_rope_head_dim
        P = self.qk_nope_head_dim

        # -- 1. Reshape inputs to [users, S, ...] --------------------------
        q_nope = q_nope.view(users, S, N, P)
        q_pe = q_pe.view(users, S, N, R)
        kv_c = kv_c_normed.view(users, S, L)
        k_pe_v = k_pe.view(users, S, 1, R)

        # -- 2. Q absorption: q_nope @ W_UK_T ------------------------------
        act_dtype = q_pe.dtype
        device = q_nope.device
        q_nope_lat = torch.einsum(
            "bsnp,npl->bsnl",
            q_nope,
            layer.W_UK_T.to(device=device),
        ).to(act_dtype)

        # -- 3. Build concatenated latent Q / K ----------------------------
        q_lat = torch.cat([q_nope_lat, q_pe], dim=-1)  # [b, S, N, L+R]
        k_lat = torch.cat([kv_c.unsqueeze(2), k_pe_v], dim=-1)  # [b, S, 1, L+R]

        if is_prefill:
            self._forward_prefill(
                q_lat,
                k_lat,
                kv_cache,
                attn_metadata,
                layer,
                S,
                users,
                act_dtype,
                device,
                output,
            )
        else:
            self._forward_decode(
                q_lat,
                k_lat,
                kv_cache,
                attn_metadata,
                layer,
                users,
                act_dtype,
                device,
                output,
            )

    # ------------------------------------------------------------------ #
    # Prefill
    # ------------------------------------------------------------------ #
    def _forward_prefill(
        self,
        q_lat: torch.Tensor,
        k_lat: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        layer: "MLAAttention",
        seq_len: int,
        users: int,
        act_dtype: torch.dtype,
        device: torch.device,
        output: Optional[torch.Tensor],
    ) -> None:
        q_for_kernel = q_lat.transpose(1, 2).contiguous()  # [b, N, S, L+R]
        k_for_kernel = k_lat.transpose(1, 2).contiguous()  # [b, 1, S, L+R]

        # Sliding window is expressed as a banded causal additive mask. This runs
        # on hardware today: tt.flash_mla_prefill accepts an arbitrary attn_mask
        # (unlike the MLA *decode* kernel, which is causal-only).
        mask = self._windowed_causal_mask(users, seq_len, device, act_dtype)

        out_lat = torch.ops.tt.flash_mla_prefill(
            query=q_for_kernel,
            key=k_for_kernel,
            head_dim_v=self.kv_lora_rank,
            value=None,
            attn_mask=mask,
            is_causal=False,
            scale=self.scale,
        )  # [b, N, S, L]

        # Attention-sink fold. flash_mla_prefill has no sink argument, so we fold
        # it here from the (pre-sink) log-sum-exp of the same windowed logits:
        #   out *= exp(lse) / (exp(lse) + exp(sink)) == sigmoid(lse - sink).
        # This is a handful of extra ops that lower to StableHLO and run on HW.
        sink = self._get_attn_sink(layer, device, torch.float32)
        if sink is not None:
            # scores[b, n, i, j] = scale * q_lat[b, n, i] . k_lat[b, j]  (single
            # latent kv head broadcast over query heads; i = query seq, j = key
            # seq). Add the same windowed-causal mask, then lse over key seq.
            k2 = k_for_kernel[:, 0].float()  # [b, S_key, L+R]
            scores = torch.einsum(
                "bnih,bjh->bnij", q_for_kernel.float(), k2
            )  # [b, N, S_query, S_key]
            scores = scores * self.scale + mask.float()
            lse = torch.logsumexp(scores, dim=-1)  # [b, N, S_query]
            factor = torch.sigmoid(lse - sink.view(1, self.num_heads, 1))
            out_lat = (out_lat.float() * factor.unsqueeze(-1)).to(out_lat.dtype)

        # Expand latent output back to v_head_dim.
        out = torch.einsum(
            "bnsl,nlv->bnsv",
            out_lat,
            layer.W_UV.to(device=device),
        ).to(
            act_dtype
        )  # [b, N, S, V]

        out = out.transpose(1, 2).reshape(
            users * seq_len, self.num_heads * self.v_head_dim
        )

        # Persist tokens into the SWA latent cache (skipped during profiling).
        if (
            attn_metadata is not None
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.numel() > 0
        ):
            k_lat_for_fill = k_lat.transpose(1, 2)  # [b, 1, S, L+R]
            fill_page_table = attn_metadata.fill_page_table
            filled_cache = kv_cache
            for batch_idx in range(users):
                filled_cache = torch.ops.tt.paged_fill_cache(
                    filled_cache,
                    k_lat_for_fill[batch_idx : batch_idx + 1],
                    fill_page_table,
                    batch_idx=torch.tensor(
                        [batch_idx], dtype=torch.int32, device=kv_cache.device
                    ),
                )
            kv_cache.copy_(filled_cache)

        output.copy_(out)

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #
    def _forward_decode(
        self,
        q_lat: torch.Tensor,
        k_lat: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        layer: "MLAAttention",
        users: int,
        act_dtype: torch.dtype,
        device: torch.device,
        output: Optional[torch.Tensor],
    ) -> None:
        """Paged SWA MLA decode on TT (one token per user, S == 1).

        Sink is folded natively by the ttnn kernel (``attention_sink``, bf16).
        The sliding window is forwarded via the native ``sliding_window`` arg
        (frontend attribute ``sliding_window_size``), which is now plumbed
        through tt-mlir to the ttnn kernel (see tt_mlir_changes.md) and applies
        on the tt device as well as in the ``cpu`` reference.
        """
        if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() > 0:
            k_lat_for_update = k_lat.transpose(0, 1)  # [1, users, 1, L+R]
            updated_cache = torch.ops.tt.paged_update_cache(
                kv_cache,
                k_lat_for_update,
                attn_metadata.cache_position,
                attn_metadata.page_table,
            )
            kv_cache.copy_(updated_cache)

        # The ttnn MLA decode kernel requires a bf16 sink tensor.
        sink = self._get_attn_sink(layer, device, torch.bfloat16)

        decode_kwargs = dict(
            query=q_lat.transpose(0, 1),  # [1, users, N, L+R]
            key=kv_cache,
            head_dim_v=self.kv_lora_rank,
            page_table=attn_metadata.page_table,
            value=None,
            is_causal=True,
            cur_pos_tensor=attn_metadata.cache_position,
            attention_sink=sink,
            scale=self.scale,
        )
        # sliding_window is plumbed through tt-mlir to the ttnn kernel (see
        # tt_mlir_changes.md); it applies on the tt device and in the cpu ref.
        if self.sliding_window is not None and self.sliding_window > 0:
            decode_kwargs["sliding_window"] = self.sliding_window

        out_lat = torch.ops.tt.paged_flash_mla_decode(**decode_kwargs)  # [1,users,N,L]

        out_lat = out_lat.reshape(users, self.num_heads, self.kv_lora_rank)
        out = torch.einsum(
            "bnl,nlv->bnv",
            out_lat,
            layer.W_UV.to(device=device),
        ).to(
            act_dtype
        )  # [users, N, V]

        out = out.reshape(users, self.num_heads * self.v_head_dim)
        output.copy_(out)


# --------------------------------------------------------------------------- #
# OOT layer wrapper (prototype)
# --------------------------------------------------------------------------- #
# vLLM's ``DeepseekV4MultiHeadLatentAttentionWrapper`` is a ``PluggableLayer``
# whose ``__init__`` is hard-bound to CUDA (asserts ``get_device_capability() is
# not None``, allocates ``torch.cuda.Event``, and its ``DeepseekV4MLAAttention``
# sub-layer asserts an fp8 kv-cache). We register an OOT replacement under the
# same key the base dispatches on (``PluggableLayer.__new__`` looks up
# ``cls.__name__`` in the OOT registry), letting our ``__init__`` run instead.
#
# Following tpu-inference's precedent, we do NOT rewrite the base ctor — we run
# it under temporary monkeypatches that neutralize the CUDA-only construction
# calls, so it still builds all the (device-agnostic) projection / norm / rope
# submodules. ``forward`` is then fully overridden to bypass the fused CUDA op.
#
# Import is guarded so this module stays importable everywhere; the class is only
# defined + registered when the upstream DSV4 layer module is present.
try:
    from vllm.model_executor.layers.deepseek_v4_attention import (
        DeepseekV4MultiHeadLatentAttentionWrapper as _DSV4Wrapper,
    )

    _DSV4_WRAPPER_AVAILABLE = True
except Exception:  # pragma: no cover - upstream DSV4 layer not present
    _DSV4_WRAPPER_AVAILABLE = False


if _DSV4_WRAPPER_AVAILABLE:

    @_DSV4Wrapper.register_oot
    class TTDeepseekV4MLAWrapper(_DSV4Wrapper):
        """TT OOT replacement for ``DeepseekV4MultiHeadLatentAttentionWrapper``.

        Constructs the CUDA-bound upstream ctor under monkeypatches (see module
        comment), then overrides ``forward``. Registered via the same
        ``register_oot`` hook the V3 ``TTMultiHeadLatentAttentionWrapper`` uses.
        """

        def __init__(self, *args, **kwargs):
            import contextlib
            from types import SimpleNamespace
            from unittest import mock

            from vllm.config import get_current_vllm_config
            from vllm.platforms import current_platform

            # cap needs a `.major` int (used for an fp8 einsum recipe we never
            # run); any value works since forward is overridden.
            dummy_cap = SimpleNamespace(major=9, minor=0)

            # torch(.cuda).Event: GPU stream-overlap events; no-op on tt.
            orig_cuda_event = torch.cuda.Event
            orig_event = getattr(torch, "Event", None)
            torch.cuda.Event = lambda *a, **k: None
            if orig_event is not None:
                torch.Event = lambda *a, **k: None

            # DeepseekV4MLAAttention asserts a fp8 kv-cache dtype at construction
            # (the cache tensor is not allocated here). Present fp8 to satisfy the
            # assert, then restore — tt uses a bf16 latent cache.
            vcfg = get_current_vllm_config()
            cache_config = getattr(vcfg, "cache_config", None)
            orig_cache_dtype = (
                getattr(cache_config, "cache_dtype", None)
                if cache_config is not None
                else None
            )
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"

            try:
                with contextlib.ExitStack() as stack:
                    # get_device_capability() is None on tt -> defeat the CUDA
                    # assert; device_type -> "cpu" so buffer allocs land on CPU
                    # (they are unused; forward is overridden).
                    stack.enter_context(
                        mock.patch.object(
                            current_platform,
                            "get_device_capability",
                            lambda *a, **k: dummy_cap,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(current_platform, "device_type", "cpu")
                    )
                    super().__init__(*args, **kwargs)
            finally:
                torch.cuda.Event = orig_cuda_event
                if orig_event is not None:
                    torch.Event = orig_event
                if cache_config is not None:
                    cache_config.cache_dtype = orig_cache_dtype

            # No quantization on the tt path — everything runs in bf16. The base
            # ctor builds an fp8 activation quantizer (QuantFP8) for the upstream
            # fp8 o-proj and leaves mla_attn tagged fp8; our bf16 forward uses
            # neither. Drop them so the wrapper carries no quantization state
            # (assumes weights are dequantized to bf16 at load time).
            if hasattr(self, "_wo_a_act_quant"):
                del self._wo_a_act_quant
            if hasattr(self, "mla_attn"):
                self.mla_attn.kv_cache_dtype = "auto"

            logger.info(
                "[TT] Constructed TTDeepseekV4MLAWrapper (prefix=%s) on tt via the "
                "monkeypatch-around-super().__init__() recipe (bf16, no quant).",
                getattr(self, "prefix", "?"),
            )

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            llama_4_scaling: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """DSV4 SWA-only forward, bf16, direct MLA (no V3 absorption).

            Reimplements the upstream fused-CUDA forward
            (``torch.ops.vllm.deepseek_v4_attention`` + fp8 o-proj) with the
            stored device-agnostic submodules:

              hidden -> fused_wqa_wkv -> split(q_lora, kv) -> q_norm / kv_norm
                     -> wq_b -> q [T,N,head_dim]; per-head Q RMSNorm
                     -> decoupled RoPE on q / kv rope dims
                     -> windowed + attention-sink MLA (V = full latent)
                     -> inverse-RoPE -> grouped o-proj (wo_a, wo_b)

            KV cache: reads the paged SWA latent cache from
            ``self.swa_cache_layer.kv_cache`` and per-layer metadata (a
            ``TTMetadata``: page_table / cache_position / fill_page_table) from
            the forward context, keyed by ``self.swa_cache_layer.prefix``. When a
            cache is bound it dispatches paged prefill (write + windowed
            self-attend) vs paged decode (write new token + windowed paged read);
            with no cache it does fresh windowed self-attention over the current
            tokens. In a full model, ``TTDeepseekV4ModelRunner`` supplies the
            cache + metadata (see get_kv_cache_spec / _prepare_inputs).

            STATUS: validated on the tt device (bf16 tt-vs-cpu PCC ~0.9998; fp32
            vs a pure-torch reference ~1.0; test_dsv4_wrapper_forward.py, and
            prefill->decode cache round-trip in test_dsv4_paged_cache.py).
            Everything is bf16 — the base ctor's fp8 quantizer (QuantFP8) is
            dropped in __init__ (weights are assumed dequantized to bf16). The
            decode sliding window is plumbed through tt-mlir to the ttnn kernel
            (tt_mlir_changes.md) and applies on the tt device.
            """
            if hidden_states.dim() != 2:
                raise NotImplementedError(
                    "TTDeepseekV4MLAWrapper.forward supports a [tokens, hidden] "
                    f"sequence; got shape {tuple(hidden_states.shape)}."
                )

            N = self.n_local_heads
            Hd = self.head_dim
            dev = hidden_states.device
            T = hidden_states.shape[0]

            q, kv, cos, sin = self._dsv4_preprocess(positions, hidden_states)
            sink = self.mla_attn.attn_sink[:N].to(device=dev)
            kv_cache, md = self._swa_cache_and_metadata()

            if md is None or not _is_bound_cache(kv_cache):
                # Fresh sequence (no paged cache): windowed self-attention.
                o = self._swa_self_attention(
                    q.view(1, T, N, Hd), kv.view(1, T, Hd), sink
                )
                o = o.reshape(T, N, Hd)
            else:
                users = md.cache_position.shape[0]
                assert T % users == 0, f"tokens ({T}) not divisible by users ({users})"
                S = T // users
                q4 = q.view(users, S, N, Hd)
                kv3 = kv.view(users, S, Hd)
                if S > 1:  # prefill (all requests have >1 token this step)
                    o = self._swa_paged_prefill(q4, kv3, kv_cache, md, sink)
                else:  # decode (one token per request)
                    o = self._swa_paged_decode(q4, kv3, kv_cache, md, sink)
                o = o.reshape(T, N, Hd)

            return self._dsv4_oproj(o, cos, sin)

        # -- helpers --------------------------------------------------------
        def _dsv4_preprocess(self, positions, hidden_states):
            """hidden -> (q [T,N,Hd] normed+roped, kv [T,Hd] normed+roped latent,
            cos, sin)."""
            T = hidden_states.shape[0]
            N, Hd, nope, rope = (
                self.n_local_heads,
                self.head_dim,
                self.nope_head_dim,
                self.rope_head_dim,
            )
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            qr, kv = qr_kv.split([self.q_lora_rank, Hd], dim=-1)
            qr = self.q_norm(qr)
            kv = self.kv_norm(kv)
            q = self.wq_b(qr).view(T, N, Hd)
            q = self.q_head_norm(q)  # per-head RMSNorm (no weight)
            cos, sin = _dsv4_cos_sin(self.rotary_emb.cos_sin_cache, positions, rope)
            q = _dsv4_rope_rope_dims(q, cos, sin, nope, headed=True)
            kv = _dsv4_rope_rope_dims(kv, cos, sin, nope, headed=False)
            return q, kv, cos, sin

        def _dsv4_oproj(self, o, cos, sin):
            """inverse-RoPE the attention output, then the grouped o-projection."""
            T = o.shape[0]
            N, Hd = self.n_local_heads, self.head_dim
            g = self.n_local_groups
            o = _dsv4_rope_rope_dims(
                o, cos, sin, self.nope_head_dim, headed=True, sign=-1.0
            )
            o_g = o.reshape(T, g, (N // g) * Hd)
            # wo_a is a grouped (bmm) ColumnParallelLinear: its weight is stored
            # 2D as [g * o_lora_rank, hpg*Hd]; reshape to [g, o_lora_rank, hpg*Hd]
            # for the per-group matmul (matches the reference modified_model
            # o-proj `wo_a.weight.view(n_local_groups, o_lora_rank, -1)`).
            wo_a_w = self.wo_a.weight.reshape(g, -1, o_g.shape[-1])
            z = torch.einsum("bhr,hdr->bhd", o_g.float(), wo_a_w.float())
            return self.wo_b(z.flatten(1).to(o.dtype))

        def _swa_cache_and_metadata(self):
            """(swa latent cache, TTMetadata) from self + the forward context."""
            kv_cache = getattr(getattr(self, "swa_cache_layer", None), "kv_cache", None)
            md = None
            try:
                from vllm.forward_context import get_forward_context

                fc = get_forward_context()
                attn_md = getattr(fc, "attn_metadata", None)
                if isinstance(attn_md, dict):
                    md = attn_md.get(self.swa_cache_layer.prefix)
                else:
                    md = attn_md
            except Exception:
                md = None
            return kv_cache, md

        def _windowed_mask(self, S, dev, dtype):
            i = torch.arange(S, device=dev)[:, None]
            j = torch.arange(S, device=dev)[None, :]
            keep = j <= i
            if self.window_size is not None and self.window_size > 0:
                keep = keep & (j > i - self.window_size)
            return torch.zeros(S, S, dtype=dtype, device=dev).masked_fill(
                ~keep, _NEG_INF
            )

        def _swa_self_attention(self, q4, kv3, sink):
            """Windowed-causal MLA attention with per-head sink, per user, over the
            current sequence. q4 [users,S,N,Hd], kv3 [users,S,Hd]; V = full latent
            (zero-pad qk to satisfy the prefill kernel's head_dim_v < qk).
            Returns o [users, S, N, Hd]."""
            users, S, N, Hd = q4.shape
            dev, dt, scale = q4.device, q4.dtype, self.scale
            mask = self._windowed_mask(S, dev, dt)  # [S,S]
            mask4 = mask.view(1, 1, S, S).expand(users, 1, S, S)
            qk = q4.permute(0, 2, 1, 3).contiguous()  # [users, N, S, Hd]
            kk = kv3.unsqueeze(1).contiguous()  # [users, 1, S, Hd]
            zq = torch.zeros(users, N, S, _MLA_V_PAD, dtype=dt, device=dev)
            zk = torch.zeros(users, 1, S, _MLA_V_PAD, dtype=dt, device=dev)
            out = torch.ops.tt.flash_mla_prefill(
                query=torch.cat([qk, zq], dim=-1),
                key=torch.cat([kk, zk], dim=-1),
                head_dim_v=Hd,
                value=None,
                attn_mask=mask4,
                is_causal=False,
                scale=scale,
            )  # [users, N, S, Hd]
            scores = torch.einsum("unih,ujh->unij", qk.float(), kk[:, 0].float())
            scores = scores * scale + mask.float()[None, None]
            lse = torch.logsumexp(scores, dim=-1)  # [users, N, S]
            factor = torch.sigmoid(lse - sink.float()[None, :, None])
            out = (out.float() * factor.unsqueeze(-1)).to(dt)  # [users, N, S, Hd]
            return out.permute(0, 2, 1, 3)  # [users, S, N, Hd]

        def _write_swa_cache_prefill(self, kv3, kv_cache, md):
            """Persist per-token latents into the paged SWA cache (prefill)."""
            users = kv3.shape[0]
            k_for_fill = kv3.unsqueeze(2).transpose(1, 2)  # [users, 1, S, Hd]
            filled = kv_cache
            for b in range(users):
                filled = torch.ops.tt.paged_fill_cache(
                    filled,
                    k_for_fill[b : b + 1],
                    md.fill_page_table,
                    batch_idx=torch.tensor(
                        [b], dtype=torch.int32, device=kv_cache.device
                    ),
                )
            kv_cache.copy_(filled)

        def _swa_paged_prefill(self, q4, kv3, kv_cache, md, sink):
            """Windowed self-attention over the current tokens + write the latents
            into the paged SWA cache. Returns o [users, S, N, Hd]."""
            o = self._swa_self_attention(q4, kv3, sink)
            self._write_swa_cache_prefill(kv3, kv_cache, md)
            return o

        def _swa_paged_decode(self, q4, kv3, kv_cache, md, sink):
            """Write the new token's latent into the paged cache, then windowed
            paged read. q4 [users,1,N,Hd], kv3 [users,1,Hd]. Returns
            o [users, 1, N, Hd]."""
            users, _, N, Hd = q4.shape
            # write new token at cache_position
            k_for_update = kv3.transpose(0, 1).unsqueeze(2)  # [1, users, 1, Hd]
            updated = torch.ops.tt.paged_update_cache(
                kv_cache, k_for_update, md.cache_position, md.page_table
            )
            kv_cache.copy_(updated)
            # windowed paged read (V = full latent; decode kernel allows hdv==qk)
            sw = (
                self.window_size
                if (self.window_size and self.window_size > 0)
                else None
            )
            decode_kwargs = dict(
                query=q4.transpose(0, 1),  # [1, users, N, Hd]
                key=kv_cache,
                head_dim_v=Hd,
                page_table=md.page_table,
                value=None,
                is_causal=True,
                cur_pos_tensor=md.cache_position,
                # Fold the sink in torch (below) rather than via the ttnn kernel's
                # native attention_sink arg. Under tensor parallelism the query
                # heads are SPMD-sharded (e.g. 64 -> 16/device) but the kernel is
                # an opaque custom call that replicates its sink operand, so its
                # sink_shape[0]==q_shape[2] / ==TILE_WIDTH checks fail ("Attention
                # sink must be a single tile wide, but got 64"). The torch fold
                # mirrors _swa_self_attention (prefill) and keeps the sink sharded
                # with the query heads.
                attention_sink=None,
                scale=self.scale,
            )
            if sw is not None:
                decode_kwargs["sliding_window"] = sw
            out = torch.ops.tt.paged_flash_mla_decode(**decode_kwargs)  # [1,users,N,Hd]
            out = out.reshape(users, 1, N, Hd)
            return self._fold_paged_decode_sink(out, q4, kv_cache, md, sink, sw)

        def _fold_paged_decode_sink(self, out, q4, kv_cache, md, sink, sw):
            """Fold per-head attention sink into the paged decode output in torch.

            Mirrors the sink fold in :meth:`_swa_self_attention` (prefill) but
            recomputes the (pre-sink) windowed log-sum-exp from the paged latent
            cache, so the sink stays sharded with the query heads instead of
            hitting the ttnn decode kernel's replicated-sink shape constraint.
            ``out`` / ``q4`` are [users, 1, N, Hd]; ``kv_cache`` [nblk, 1, blk, Hd].
            """
            users, _, N, Hd = q4.shape
            dev = q4.device
            block_size = kv_cache.shape[-2]
            # Gather each user's windowed K from the paged latent cache (page_table
            # lists physical block ids in logical order).
            gk = kv_cache[md.page_table.long()]  # [users, nbpu, 1, blk, Hd]
            max_seq = gk.shape[1] * block_size
            gk = gk.permute(0, 2, 1, 3, 4).reshape(users, max_seq, Hd).float()
            q = q4.reshape(users, N, Hd).float()
            scores = torch.einsum("und,usd->uns", q, gk) * self.scale  # [users,N,S]
            # Causal + sliding-window mask from cache_position (traceable).
            j = torch.arange(max_seq, device=dev).view(1, 1, max_seq)
            cur = md.cache_position.view(users, 1, 1)
            keep = j <= cur
            if sw is not None and sw > 0:
                keep = keep & (j > cur - sw)
            scores = torch.where(keep, scores, torch.full_like(scores, _NEG_INF))
            lse = torch.logsumexp(scores, dim=-1)  # [users, N]
            factor = torch.sigmoid(lse - sink.float().view(1, N))
            return (out.float() * factor.view(users, 1, N, 1)).to(out.dtype)


# ============================================================================ #
# What's next (out of scope for the SWA-only milestone)
# ---------------------------------------------------------------------------- #
# 1. Decode sliding window on hardware: unblock `slidingWindowSize` in the
#    tt-mlir runtime executor `paged_flash_multi_latent_attention_decode.cpp`
#    (currently `std::nullopt`) and thread a `sliding_window_size` attribute
#    through StableHLOToTTIR / TTIR / TTNN / flatbuffer — cloning the fully
#    plumbed `PagedScaledDotProductAttentionDecodeOp` sibling. Diff in the
#    next-steps doc.
# 2. C128A (compress_ratio == 128): a contiguous compressed-prefix branch of
#    length `(pos + 1) // ratio` plus the window branch, merged via online
#    softmax. The compressed branch is dense MLA attention over the prefix — it
#    can reuse tt.flash_mla_prefill / tt.paged_flash_mla_decode, no sparse kernel
#    needed. A second KV cache group holds the compressed latent.
# 3. C4A (compress_ratio == 4): the lightning-indexer top-k branch. NOTE: the
#    pinned tt-metal ships NO sparse SDPA kernel (`sparse_sdpa` does not exist),
#    so the sparse gather/attention has no kernel — this needs a new tt-metal
#    kernel (or a gather-then-dense fallback) before it can run on hardware.
# 4. Two-branch online-softmax merge: combine the window-branch and
#    compressed-branch outputs. Because the MLA decode op returns no LSE, either
#    (a) gather both index sets into one workspace and run a single attention
#    call (mirrors the GPU prefill path), or (b) add an LSE output to the ops and
#    merge in StableHLO. Prototype against the CPU reference before committing.
# 5. Full-model wiring: a `PluggableLayer.register("deepseek_v4_multi_head_
#    latent_attention")` TT wrapper (the upstream wrapper is CUDA-only), plus
#    platform gating (unblock use_sparse for DSV4), multi-group KV cache
#    (initialize_kv_cache rejects >1 group), and per-branch metadata (the
#    single-`TTMetadata` `dict.fromkeys` fan-out). See platform.py / model_runner.
# ============================================================================ #
