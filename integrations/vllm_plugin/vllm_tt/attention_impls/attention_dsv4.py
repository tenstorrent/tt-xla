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
# Large *finite* mask fill for attention branches that can have a fully-masked
# row (a query with zero valid compressed slots). torch.logsumexp / softmax of an
# all-`-inf` row returns nan on XLA (it computes exp(-inf - (-inf)) = exp(nan)),
# unlike torch on CPU which special-cases it. A finite fill is numerically
# identical when >=1 entry is valid (exp(-1e30 - max) == 0), and for an all-masked
# row yields a finite, very-negative lse that the two-branch merge zeroes out
# (exp(lse_c - m) -> 0). The window branch never has an all-masked row (a query
# always attends itself), so it keeps _NEG_INF.
_MASK_NEG = -1e30

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
        # NOTE: no compress_ratio guard here. This impl is dormant — the active
        # DSV4 attention runs through TTDeepseekV4MLAWrapper.forward, which
        # dispatches SWA / C128A and raises for C4A. Constructing this (possibly
        # instantiated by DeepseekV4MLAAttention) must succeed for every layer
        # type so the model can build; its forward_mha/forward_mqa are hard stubs.

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


# The DeepseekV4Indexer (the C4A lightning indexer) is constructed inside
# ``DeepseekV4Attention.__init__`` (the model file), OUTSIDE the
# ``TTDeepseekV4MLAWrapper`` ctor below — so it does NOT run under that ctor's
# ``device_type="cpu"`` monkeypatch. Its own ``DeepseekCompressor`` allocates the
# ``ape`` Parameter on ``current_platform.device_type`` explicitly (== "xla" on
# tt), while every sibling parameter is built on CPU (the model is constructed
# under ``load_config.device="cpu"``). That device mismatch breaks weight loading
# / ``model.to(xla)``. Force the indexer ctor to run with ``device_type="cpu"``
# (mirrors the wrapper recipe; only ``device_type`` is load-bearing —
# ``SparseAttnIndexer.__init__`` only asserts ``is_cuda()``, which is False on tt).
try:
    from vllm.model_executor.layers.deepseek_v4_attention import (
        DeepseekV4Indexer as _DSV4Indexer,
    )

    _dsv4_orig_indexer_init = _DSV4Indexer.__init__

    def _tt_dsv4_indexer_init(self, *args, **kwargs):
        from unittest import mock

        from vllm.platforms import current_platform

        with mock.patch.object(current_platform, "device_type", "cpu"):
            _dsv4_orig_indexer_init(self, *args, **kwargs)

    _DSV4Indexer.__init__ = _tt_dsv4_indexer_init
except Exception:  # pragma: no cover - upstream DSV4 indexer not present
    pass


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

            q, kv, cos, sin, qr = self._dsv4_preprocess(positions, hidden_states)
            sink = self.mla_attn.attn_sink[:N].to(device=dev)
            kv_cache, md = self._swa_cache_and_metadata()
            r = self.compress_ratio

            if r == 128:
                # C128A (HCA): window + dense compressed-prefix branch, merged.
                o = self._c128a_forward(
                    positions, hidden_states, q, kv, sink, kv_cache, md
                )
            elif r == 4:
                # C4A (CSA): window + lightning-indexer top-k sparse compressed
                # branch, merged. Needs qr (pre-wq_b low-rank query) for the indexer.
                o = self._c4a_forward(
                    positions, hidden_states, q, kv, qr, sink, kv_cache, md
                )
            elif r > 1:
                raise NotImplementedError(
                    f"DSV4 compress_ratio={r} is not supported on TT "
                    "(only SWA <= 1, C4A == 4, C128A == 128)."
                )
            elif md is None or not _is_bound_cache(kv_cache):
                # SWA-only, fresh sequence (no paged cache): windowed self-attention.
                o = self._swa_self_attention(
                    q.view(1, T, N, Hd), kv.view(1, T, Hd), sink
                ).reshape(T, N, Hd)
            else:
                # SWA-only, paged.
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
            cos, sin, qr [T,q_lora_rank] the pre-wq_b low-rank query). ``qr`` is
            returned so the C4A lightning indexer can reuse it (it applies its own
            wq_b to qr) without recomputing fused_wqa_wkv + q_norm."""
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
            return q, kv, cos, sin, qr

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

        def _windowed_branch(self, q4, kv3):
            """Windowed-causal MLA attention over the current tokens, WITHOUT the
            sink fold. q4 [users,S,N,Hd], kv3 [users,S,Hd]; V = full latent
            (zero-pad qk to satisfy the prefill kernel's head_dim_v < qk).
            Returns (o_w [users,N,S,Hd] kernel layout, lse_w [users,N,S]) — the
            pre-sink windowed output + its log-sum-exp, for _two_branch_merge."""
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
            return out, lse

        def _two_branch_merge(self, o_w, lse_w, o_c, lse_c, sink):
            """Combine a window branch (o_w, lse_w) and an optional compressed
            branch (o_c, lse_c) under one softmax whose denominator carries the
            per-head attention sink as an extra logit — the reference
            ``sparse_attn`` merge (modified_model/kernel.py:82-88), written as a
            numerically stable online-softmax. All tensor args are in the kernel
            layout ([users,N,S,Hd] for o_*, [users,N,S] for lse_*); ``sink`` is
            [N]. ``o_c=None`` (SWA-only) collapses to the exact
            ``sigmoid(lse_w - sink)`` fold. fp32 accumulation throughout.
            """
            sink_ = sink.float().view(1, -1, 1)  # [1, N, 1]
            ow, lw = o_w.float(), lse_w.float()
            if o_c is None:
                factor = torch.sigmoid(lw - sink_)  # [users, N, S]
                return (ow * factor.unsqueeze(-1)).to(o_w.dtype)
            oc, lc = o_c.float(), lse_c.float()
            m = torch.maximum(torch.maximum(lw, lc), sink_)  # [users, N, S]
            ew, ec, es = torch.exp(lw - m), torch.exp(lc - m), torch.exp(sink_ - m)
            denom = (ew + ec + es).unsqueeze(-1)  # [users, N, S, 1]
            o = (ow * ew.unsqueeze(-1) + oc * ec.unsqueeze(-1)) / denom
            return o.to(o_w.dtype)

        def _swa_self_attention(self, q4, kv3, sink):
            """SWA-only windowed attention with the sink fold (used by the fresh
            and paged-prefill SWA paths). Returns o [users, S, N, Hd]."""
            o_w, lse_w = self._windowed_branch(q4, kv3)
            o = self._two_branch_merge(o_w, lse_w, None, None, sink)
            return o.permute(0, 2, 1, 3)  # [users, S, N, Hd]

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

        def _paged_lse(self, q4, cache, page_table, cur_pos, sw):
            """(pre-sink) log-sum-exp of a paged single-latent-KV attention, from
            the cache. ``q4`` [users,1,N,Hd]; ``cache`` [nblk,1,blk,Hd]; ``cur_pos``
            [users] is the last valid slot per user; ``sw`` (or None) applies a
            sliding window on top of the ``j <= cur_pos`` causal mask. Returns
            lse [users, N]. Shared by the SWA sink fold and the C128A two-branch
            merge (window branch: sw=window_size; compressed branch: sw=None)."""
            users, _, N, Hd = q4.shape
            dev = q4.device
            block_size = cache.shape[-2]
            gk = cache[page_table.long()]  # [users, nbpu, 1, blk, Hd]
            max_seq = gk.shape[1] * block_size
            gk = gk.permute(0, 2, 1, 3, 4).reshape(users, max_seq, Hd).float()
            q = q4.reshape(users, N, Hd).float()
            scores = torch.einsum("und,usd->uns", q, gk) * self.scale  # [users,N,S]
            j = torch.arange(max_seq, device=dev).view(1, 1, max_seq)
            cur = cur_pos.view(users, 1, 1)
            keep = j <= cur
            if sw is not None and sw > 0:
                keep = keep & (j > cur - sw)
            # Finite fill (see _MASK_NEG): a compressed-branch query may have zero
            # valid slots; an all-`-inf` row would make logsumexp nan on XLA.
            scores = torch.where(keep, scores, torch.full_like(scores, _MASK_NEG))
            return torch.logsumexp(scores, dim=-1)  # [users, N]

        def _fold_paged_decode_sink(self, out, q4, kv_cache, md, sink, sw):
            """Fold per-head attention sink into the paged (SWA) decode output in
            torch. Mirrors the sink fold in :meth:`_swa_self_attention` but
            recomputes the (pre-sink) windowed log-sum-exp from the paged cache,
            so the sink stays sharded with the query heads instead of hitting the
            ttnn decode kernel's replicated-sink shape constraint. ``out`` /
            ``q4`` are [users, 1, N, Hd]; ``kv_cache`` [nblk, 1, blk, Hd]."""
            users, _, N, Hd = q4.shape
            lse = self._paged_lse(q4, kv_cache, md.page_table, md.cache_position, sw)
            factor = torch.sigmoid(lse - sink.float().view(1, N))
            return (out.float() * factor.view(users, 1, N, 1)).to(out.dtype)

        # -- C128A (HCA) compressed-attention branch ------------------------
        def _dsv4_compress(
            self, hidden_states, comp=None, head_dim=None, rope_cache=None
        ):
            """bf16 KV compressor over a single ``[S, hidden]`` sequence: pool every
            ``compress_ratio`` tokens into one compressed latent (gated softmax pool
            + learned APE + RMSNorm + GPT-J RoPE at the compressed position
            ``slot*ratio``). Reimplements the oracle ``Compressor.forward``
            (modified_model/model.py:364-450, start_pos==0) for BOTH the non-overlap
            case (C128A: coff==1) and the overlap case (C4A: coff==2). Parameterized
            over the compressor module so it serves the main compressor
            (``head_dim`` = self.head_dim) AND the indexer's own compressor
            (``head_dim`` = index_head_dim). Returns ``comp [Ncomp, head_dim]`` with
            ``Ncomp = S // ratio``, or ``None`` for a sub-window chunk (Ncomp==0).
            Trailing ``S % ratio`` tokens are deferred (rolling compressor state)."""
            comp = self.compressor if comp is None else comp
            d = self.head_dim if head_dim is None else head_dim
            rope_cache = (
                self.rotary_emb.cos_sin_cache if rope_cache is None else rope_cache
            )
            ratio, coff, overlap = comp.compress_ratio, comp.coff, comp.overlap
            # nope MUST be derived from the PASSED head_dim (64 for the index
            # compressor's d=128, 448 for the main d=512) — NOT self.nope_head_dim.
            nope, rope = d - self.rope_head_dim, self.rope_head_dim
            w = coff * d
            dt, dev = hidden_states.dtype, hidden_states.device
            S = hidden_states.shape[0]
            # Sub-window chunk (Ncomp == 0, e.g. the 32/64-token prefill compile
            # buckets): NO compressed group completes. Return None rather than build
            # a 0-sized tensor (a 0 dim divides-by-zero in tt-mlir layout analysis);
            # the chunk is genuinely window+sink only. `S` is a trace-time constant,
            # so this prunes the compressed subgraph for those buckets entirely.
            if S // ratio == 0:
                return None
            # fused_wkv_wgate is return_bias=False -> bare tensor; tolerate a tuple.
            kvsc = comp.fused_wkv_wgate(hidden_states)  # [S, 2*coff*d]
            if isinstance(kvsc, (tuple, list)):
                kvsc = kvsc[0]
            kv, score = kvsc.split([w, w], dim=-1)  # each [S, coff*d]
            Ncomp = S // ratio
            cutoff = Ncomp * ratio
            kv = kv[:cutoff].reshape(Ncomp, ratio, w).float()
            score = (
                score[:cutoff].reshape(Ncomp, ratio, w).float() + comp.ape.float()
            )  # ape [ratio, coff*d] broadcasts over (Ncomp, ratio)
            if overlap:
                # oracle overlap_transform (model.py:356-362): each completed slot
                # pools 2*ratio entries = [previous block's first-half channels
                # (shifted +1 slot), current block's second-half channels]. slot 0's
                # previous half is a fill (0 for kv, _MASK_NEG score -> zero weight).
                kv_prev, kv_cur = kv[..., :d], kv[..., d:]  # each [Ncomp, ratio, d]
                sc_prev, sc_cur = score[..., :d], score[..., d:]
                kv = torch.cat(
                    [
                        torch.cat([kv.new_zeros(1, ratio, d), kv_prev[:-1]], dim=0),
                        kv_cur,
                    ],
                    dim=1,
                )  # [Ncomp, 2*ratio, d]
                score = torch.cat(
                    [
                        torch.cat(
                            [score.new_full((1, ratio, d), _MASK_NEG), sc_prev[:-1]],
                            dim=0,
                        ),
                        sc_cur,
                    ],
                    dim=1,
                )
            pooled = (kv * score.softmax(dim=1)).sum(dim=1)  # [Ncomp, d] gated pool
            pooled = comp.norm(pooled.to(dt))  # RMSNorm(d)
            pos_c = torch.arange(Ncomp, device=dev) * ratio
            cos_c, sin_c = _dsv4_cos_sin(rope_cache, pos_c, rope)
            return _dsv4_rope_rope_dims(pooled, cos_c, sin_c, nope, headed=False)

        def _dsv4_index_compress(self, hidden_states):
            """The lightning indexer's OWN overlap compressor (head_dim=128) ->
            compressed KEYS ``[Ncomp, 128]``. Same overlap math as the main
            compressor, but a different module, head_dim, and RoPE cache."""
            ix = self.indexer
            return self._dsv4_compress(
                hidden_states,
                comp=ix.compressor,
                head_dim=ix.head_dim,
                rope_cache=self.indexer_rotary_emb.cos_sin_cache,
            )

        def _dsv4_indexer(self, qr, hidden_states, positions):
            """DSV4 C4A lightning indexer (oracle ``Indexer.forward``,
            modified_model/model.py:487-521) over one ``[S, hidden]`` sequence:
            score every compressed slot with the per-head indexer query against the
            indexer's own compressed keys (relu per head, gate, sum over heads), then
            top-k select. Returns ``topk_idxs [S, k]`` (k = min(index_topk, Ncomp))
            into ``[0, Ncomp)`` with causally-invalid picks == -1, or ``None`` for a
            sub-window chunk (Ncomp==0). NO ``k_norm`` (the vLLM indexer constructs
            but never applies it; not in the oracle). ``qr`` is the pre-wq_b low-rank
            query from ``_dsv4_preprocess``."""
            ix = self.indexer
            ratio = self.compress_ratio
            nh, hd, rope = ix.n_head, ix.head_dim, self.rope_head_dim  # 64, 128, 64
            S = hidden_states.shape[0]
            Kc = self._dsv4_index_compress(hidden_states)  # [Ncomp, 128] or None
            if Kc is None:
                return None
            Ncomp = Kc.shape[0]
            # per-head indexer query: wq_b(qr) -> [S, nh, hd], GPT-J RoPE last `rope`.
            q, _ = ix.wq_b(qr)
            q = q.view(S, nh, hd)
            cos, sin = _dsv4_cos_sin(
                self.indexer_rotary_emb.cos_sin_cache, positions, rope
            )
            q = _dsv4_rope_rope_dims(q, cos, sin, hd - rope, headed=True).float()
            # per-(token, head) gate; oracle folds softmax_scale * n_head**-0.5.
            weights, _ = ix.weights_proj(hidden_states)
            weights = weights.float() * (ix.softmax_scale * nh**-0.5)  # [S, nh]
            # index score = relu(q . Kc) per head, gate, sum over heads -> [S, Ncomp].
            # Kc is shared across heads, so fold the einsum into a single 2D matmul
            # [S*nh, hd] @ [hd, Ncomp] — the "shd,td->sht" form would materialize a
            # [S, nh, Ncomp, hd] broadcast intermediate on tt-mlir (OOM at ratio=4).
            sc = torch.mm(q.reshape(S * nh, hd), Kc.float().t())  # [S*nh, Ncomp]
            score = sc.view(S, nh, Ncomp).relu()  # [S, nh, Ncomp]
            score = (score * weights.unsqueeze(-1)).sum(dim=1)  # [S, Ncomp]
            # causal mask: slot c valid for query at position p iff c < (p+1)//ratio.
            # _MASK_NEG (finite) so all-masked early rows (p < ratio-1) don't nan
            # topk/softmax on XLA.
            valid_ct = ((positions + 1) // ratio).view(S, 1)  # [S, 1]
            c_idx = torch.arange(Ncomp, device=Kc.device).view(1, Ncomp)
            score = torch.where(
                c_idx < valid_ct, score, torch.full_like(score, _MASK_NEG)
            )
            k = min(ix.topk_tokens, Ncomp)  # trace-time constant
            topk_idxs = score.topk(k, dim=-1)[1]  # [S, k] int64
            # re-mask causally-invalid picks (NO +offset — the sparse branch gathers
            # directly into comp[0..Ncomp)).
            keep = topk_idxs < valid_ct
            topk_idxs = torch.where(keep, topk_idxs, torch.full_like(topk_idxs, -1))
            return topk_idxs.int()

        def _compressed_branch(self, q4, comp, pos2d):
            """Dense MLA attention over ALL valid compressed latents (C128A). Query
            at absolute position p attends compressed slot c iff ``c < (p+1)//ratio``
            (causal-over-compressed, oracle ``get_compress_topk_idxs``); V = the
            full compressed latent. ``q4`` [users,S,N,Hd], ``comp`` [users,Ncomp,Hd],
            ``pos2d`` [users,S] absolute positions. Returns (o_c [users,N,S,Hd],
            lse_c [users,N,S]); fully-masked rows give o_c=0, lse_c=-inf (the merge
            zeroes their contribution)."""
            users, S, N, Hd = q4.shape
            dev, scale, ratio = q4.device, self.scale, self.compress_ratio
            Ncomp = comp.shape[1]
            qk = q4.permute(0, 2, 1, 3).float()  # [users, N, S, Hd]
            compf = comp.float()  # [users, Ncomp, Hd]
            scores = torch.einsum("unih,uch->unic", qk, compf) * scale  # [u,N,S,Nc]
            valid = ((pos2d + 1) // ratio).view(users, 1, S, 1)  # slots c < valid
            c_idx = torch.arange(Ncomp, device=dev).view(1, 1, 1, Ncomp)
            keep = c_idx < valid  # [users,1,S,Ncomp] -> broadcasts over heads
            # Finite mask fill (see _MASK_NEG): all-masked rows -> finite very-neg
            # lse (merge zeroes them) and uniform (finite) softmax, avoiding the
            # XLA all-`-inf` logsumexp/softmax nan.
            scores = torch.where(keep, scores, torch.full_like(scores, _MASK_NEG))
            lse_c = torch.logsumexp(scores, dim=-1)  # [users, N, S] (very-neg if empty)
            w = torch.softmax(scores, dim=-1)
            o_c = torch.einsum("unic,uch->unih", w, compf)  # [users, N, S, Hd]
            return o_c.to(q4.dtype), lse_c

        def _sparse_compressed_branch(self, q4, comp, topk_idxs):
            """C4A SPARSE MLA attention over the per-query top-k selected compressed
            latents (torch port of oracle ``sparse_attn`` kernel.py:40-96, sink-free
            — the sink is deferred to ``_two_branch_merge``). ``q4`` [users,S,N,Hd],
            ``comp`` [users,Ncomp,Hd] (the main compressor VALUES), ``topk_idxs``
            [users,S,k] into [0,Ncomp) with -1 for invalid picks (from
            ``_dsv4_indexer``). Gathers the selected slots per query (1D index_select,
            the tt-mlir-friendly form) and attends. Returns (o_c [users,N,S,Hd],
            lse_c [users,N,S]) in the SAME layout as ``_compressed_branch`` so the
            merge is reused unchanged. Fully-masked rows -> lse_c very-neg (merge
            zeroes them). NOTE: for seqlen <= index_topk*ratio (=2048) the indexer
            selects ALL valid slots, so this is numerically identical to the dense
            ``_compressed_branch`` (softmax is permutation-invariant)."""
            users, S, N, Hd = q4.shape
            Ncomp, k = comp.shape[1], topk_idxs.shape[-1]
            scale = self.scale
            valid = topk_idxs != -1  # [users, S, k]
            safe = topk_idxs.clamp(min=0).long()
            # 1D gather: flatten comp to [users*Ncomp, Hd], offset per user.
            comp_flat = comp.reshape(users * Ncomp, Hd)
            b_off = (torch.arange(users, device=comp.device) * Ncomp).view(users, 1, 1)
            flat = (safe + b_off).reshape(-1)  # [users*S*k]
            gathered = (
                torch.index_select(comp_flat, 0, flat).reshape(users, S, k, Hd).float()
            )  # [users, S, k, Hd]
            # scores + o_c as batched matmuls over batch=(users*S) — the disjoint
            # einsums "unsh,uskh->unsk" / "unsk,uskh->unsh" materialize a 5D
            # [users, N, S, k, Hd] broadcast on tt-mlir (~2GB fp32 at ratio=4;
            # OOM), whereas the bmm keeps every intermediate <= [users*S, N, Hd].
            # q4 [u,S,N,Hd] and gathered [u,S,k,Hd] flatten contiguously to (u*S).
            B = users * S
            qk_b = q4.reshape(B, N, Hd).float()  # [B, N, Hd]
            g_b = gathered.reshape(B, k, Hd)  # [B, k, Hd]
            scores = torch.bmm(qk_b, g_b.transpose(1, 2)) * scale  # [B, N, k]
            scores = scores.view(users, S, N, k).permute(0, 2, 1, 3)  # [u, N, S, k]
            # mask invalid (-1) picks (broadcast over heads) with finite _MASK_NEG.
            scores = torch.where(
                valid.view(users, 1, S, k),
                scores,
                torch.full_like(scores, _MASK_NEG),
            )
            lse_c = torch.logsumexp(scores, dim=-1)  # [users, N, S]
            w = torch.softmax(scores, dim=-1)  # [users, N, S, k]
            o_b = torch.bmm(w.permute(0, 2, 1, 3).reshape(B, N, k), g_b)  # [B, N, Hd]
            o_c = o_b.view(users, S, N, Hd).permute(0, 2, 1, 3)  # [u, N, S, Hd]
            return o_c.to(q4.dtype), lse_c

        def _compressed_cache_and_metadata(self):
            """(compressed latent cache, TTMetadata) for the C128A/C4A compressed
            group — keyed by the DeepseekV4MLAAttention (``mla_attn``) prefix, the
            second cache group emitted by the model_runner."""
            comp_cache = getattr(getattr(self, "mla_attn", None), "kv_cache", None)
            md = None
            try:
                from vllm.forward_context import get_forward_context

                fc = get_forward_context()
                attn_md = getattr(fc, "attn_metadata", None)
                if isinstance(attn_md, dict):
                    md = attn_md.get(self.mla_attn.prefix)
                else:
                    md = attn_md
            except Exception:
                md = None
            return comp_cache, md

        def _write_compressed_cache_prefill(self, comp, comp_cache, comp_md):
            """Persist compressed latents into the paged compressed cache (prefill),
            mirroring :meth:`_write_swa_cache_prefill`. ``comp`` [users, Ncomp, Hd]."""
            users = comp.shape[0]
            k_for_fill = comp.unsqueeze(2).transpose(1, 2)  # [users, 1, Ncomp, Hd]
            filled = comp_cache
            for b in range(users):
                filled = torch.ops.tt.paged_fill_cache(
                    filled,
                    k_for_fill[b : b + 1],
                    comp_md.fill_page_table,
                    batch_idx=torch.tensor(
                        [b], dtype=torch.int32, device=comp_cache.device
                    ),
                )
            comp_cache.copy_(filled)

        def _c128a_forward(self, positions, hidden_states, q, kv, sink, kv_cache, md):
            """DSV4 C128A (HCA): window branch + dense compressed-prefix branch,
            merged under one softmax with the sink in the denominator. Returns
            o [T, N, Hd] (pre inverse-RoPE / o-proj). Fresh-sequence path (no
            cache) is the M1-validated path; paged prefill/decode use the second
            (compressed) cache group + per-group metadata from the model_runner."""
            N, Hd = self.n_local_heads, self.head_dim
            T = hidden_states.shape[0]
            if md is None or not _is_bound_cache(kv_cache):
                comp = self._dsv4_compress(hidden_states)  # [Ncomp,Hd] or None
                o_w, lse_w = self._windowed_branch(
                    q.view(1, T, N, Hd), kv.view(1, T, Hd)
                )
                if comp is None:
                    # sub-window chunk: window+sink only (no compressed slots yet)
                    o_c, lse_c = None, None
                else:
                    o_c, lse_c = self._compressed_branch(
                        q.view(1, T, N, Hd), comp.unsqueeze(0), positions.view(1, T)
                    )
                o = self._two_branch_merge(o_w, lse_w, o_c, lse_c, sink)  # [1,N,T,Hd]
                return o.permute(0, 2, 1, 3).reshape(T, N, Hd)
            users = md.cache_position.shape[0]
            assert T % users == 0, f"tokens ({T}) not divisible by users ({users})"
            S = T // users
            q4 = q.view(users, S, N, Hd)
            kv3 = kv.view(users, S, Hd)
            comp_cache, comp_md = self._compressed_cache_and_metadata()
            if S > 1:
                o = self._c128a_paged_prefill(
                    q4,
                    kv3,
                    hidden_states,
                    positions,
                    sink,
                    kv_cache,
                    md,
                    comp_cache,
                    comp_md,
                )
            else:
                o = self._c128a_paged_decode(
                    q4,
                    kv3,
                    sink,
                    kv_cache,
                    md,
                    comp_cache,
                    comp_md,
                    hidden_states,
                    positions,
                )
            return o.reshape(T, N, Hd)

        def _c128a_paged_prefill(
            self,
            q4,
            kv3,
            hidden_states,
            positions,
            sink,
            kv_cache,
            md,
            comp_cache,
            comp_md,
        ):
            """Paged C128A prefill: window self-attend + dense compressed attend,
            merge, then write both the SWA and compressed caches. Returns
            o [users, S, N, Hd]."""
            users, S, N, Hd = q4.shape
            hs = hidden_states.view(users, S, -1)
            pos2d = positions.view(users, S)
            o_w, lse_w = self._windowed_branch(q4, kv3)
            self._write_swa_cache_prefill(kv3, kv_cache, md)
            # Seed the compressor rolling state for exact-across-boundary decode.
            # MUST run even for a sub-window prefill (S < ratio, common for C128A
            # where ratio=128): those tokens have no completed slot yet but must be
            # in the rolling state so the first slot completes correctly at decode.
            self._seed_compressor_state(
                hs, self.compressor, self.head_dim, md.cache_position + 1
            )
            # Sub-window prefill chunk (S < ratio): no compressed slots complete,
            # so this is window+sink only. `S` is a trace-time constant, so the
            # 32/64-token compile buckets prune the compressed subgraph here
            # (avoids the 0-sized Ncomp tensor that divides-by-zero in tt-mlir).
            if S // self.compress_ratio == 0:
                o = self._two_branch_merge(o_w, lse_w, None, None, sink)
                return o.permute(0, 2, 1, 3)
            comp = torch.stack(
                [self._dsv4_compress(hs[b]) for b in range(users)], dim=0
            )  # [users, Ncomp, Hd]
            o_c, lse_c = self._compressed_branch(q4, comp, pos2d)
            o = self._two_branch_merge(o_w, lse_w, o_c, lse_c, sink)  # [users,N,S,Hd]
            if (
                _is_bound_cache(comp_cache)
                and comp_md is not None
                and comp.shape[1] > 0
            ):
                self._write_compressed_cache_prefill(comp, comp_cache, comp_md)
            return o.permute(0, 2, 1, 3)  # [users, S, N, Hd]

        def _c128a_paged_decode(
            self,
            q4,
            kv3,
            sink,
            kv_cache,
            md,
            comp_cache,
            comp_md,
            hidden_states=None,
            positions=None,
        ):
            """Paged C128A/C4A decode: window paged read (as SWA, sink-free) + a
            DENSE compressed paged read, merged. EXACT ACROSS COMPRESSION BOUNDARIES:
            when ``hidden_states``/``positions`` are given and the main compressor's
            rolling-state buffer is bound, this runs the compressor incrementally
            (`_compress_decode_step`) and writes the newly-completed slot into the
            compressed VALUES cache BEFORE the read, so the read no longer misses the
            slots that form during decode. If those are omitted / the state is not
            bound, it degrades to the legacy folded-state read (frozen prefill cache).
            Returns o [users, 1, N, Hd]."""
            users, _, N, Hd = q4.shape
            # window branch: write new token, sink-free paged read, recompute lse
            k_for_update = kv3.transpose(0, 1).unsqueeze(2)  # [1, users, 1, Hd]
            kv_cache.copy_(
                torch.ops.tt.paged_update_cache(
                    kv_cache, k_for_update, md.cache_position, md.page_table
                )
            )
            # incremental main-compressor slot formation + write into the compressed
            # VALUES cache at slot pos//ratio (the reader's cache_position =
            # (pos+1)//ratio-1 excludes the still-in-progress slot until it completes).
            if (
                hidden_states is not None
                and positions is not None
                and _is_bound_cache(comp_cache)
                and comp_md is not None
                and self._compressor_state_buf(self.compressor) is not None
            ):
                slot, wslot = self._compress_decode_step(
                    hidden_states.view(users, -1),
                    positions.view(users),
                    self.compressor,
                    self.head_dim,
                    self.rotary_emb.cos_sin_cache,
                )
                cfill = slot.unsqueeze(1).transpose(0, 1).unsqueeze(2)  # [1,users,1,Hd]
                comp_cache.copy_(
                    torch.ops.tt.paged_update_cache(
                        comp_cache,
                        cfill.to(comp_cache.dtype),
                        wslot,
                        comp_md.page_table,
                    )
                )
            sw = (
                self.window_size
                if (self.window_size and self.window_size > 0)
                else None
            )
            wk = dict(
                query=q4.transpose(0, 1),
                key=kv_cache,
                head_dim_v=Hd,
                page_table=md.page_table,
                value=None,
                is_causal=True,
                cur_pos_tensor=md.cache_position,
                attention_sink=None,
                scale=self.scale,
            )
            if sw is not None:
                wk["sliding_window"] = sw
            o_w = torch.ops.tt.paged_flash_mla_decode(**wk).reshape(users, 1, N, Hd)
            lse_w = self._paged_lse(q4, kv_cache, md.page_table, md.cache_position, sw)
            # compressed branch (no window, no sink). comp_md.cache_position is the
            # last valid compressed slot = (pos+1)//ratio - 1.
            if not (_is_bound_cache(comp_cache) and comp_md is not None):
                # No compressed slots yet (e.g. pos < ratio) -> window-only.
                o = self._two_branch_merge(
                    o_w.permute(0, 2, 1, 3), lse_w.unsqueeze(-1), None, None, sink
                )
                return o.permute(0, 2, 1, 3)
            # No-completed-slot guard (tensor-level, graph-safe). When seq_len <
            # ratio (e.g. ratio=128, the first ~127 decode steps), the reader's
            # compressed cache_position = (pos+1)//ratio - 1 is -1: there is no
            # completed compressed slot to attend to. The paged decode graph is
            # compiled once with a fixed structure, so we can't host-branch this
            # away per step; instead we neutralize the branch with tensors:
            #   * clamp the *kernel* read position to >= 0 so it reads a valid
            #     slot (0) instead of an out-of-range one -> o_c is finite, not
            #     the NaN a negative/OOB slot read would yield (NaN*0 = NaN in
            #     the softmax merge would poison the output -> empty context/BOS);
            #   * feed the *true* (unclamped) position to _paged_lse so its causal
            #     mask (j <= cur_pos) drops every slot -> lse_c ~ -inf -> the
            #     merge gives the compressed branch zero weight (window-only);
            #   * where() any residual non-finite o_c to 0 as belt-and-suspenders
            #     (0 weight * 0 value stays 0 regardless of cache init).
            # For valid positions (>= 0) all three are exact no-ops, so the
            # steady-state C128A path and the C4A path are unchanged.
            comp_valid = (comp_md.cache_position >= 0).view(users, 1, 1, 1)
            o_c = torch.ops.tt.paged_flash_mla_decode(
                query=q4.transpose(0, 1),
                key=comp_cache,
                head_dim_v=Hd,
                page_table=comp_md.page_table,
                value=None,
                is_causal=True,
                cur_pos_tensor=comp_md.cache_position.clamp(min=0),
                attention_sink=None,
                scale=self.scale,
            ).reshape(users, 1, N, Hd)
            o_c = torch.where(comp_valid, o_c, torch.zeros_like(o_c))
            lse_c = self._paged_lse(
                q4, comp_cache, comp_md.page_table, comp_md.cache_position, None
            )
            o = self._two_branch_merge(
                o_w.permute(0, 2, 1, 3),
                lse_w.unsqueeze(-1),
                o_c.permute(0, 2, 1, 3),
                lse_c.unsqueeze(-1),
                sink,
            )  # [users, N, 1, Hd]
            return o.permute(0, 2, 1, 3)  # [users, 1, N, Hd]

        def _c4a_forward(self, positions, hidden_states, q, kv, qr, sink, kv_cache, md):
            """DSV4 C4A (CSA): window branch + lightning-indexer top-k SPARSE
            compressed branch, merged under one softmax with the sink in the
            denominator. Returns o [T, N, Hd] (pre inverse-RoPE / o-proj). The
            fresh-sequence path (no cache) needs only the indexer + main compressor
            + sparse branch (validated against the oracle); paged prefill/decode add
            the 3-group KV cache + rolling compressor state."""
            N, Hd = self.n_local_heads, self.head_dim
            T = hidden_states.shape[0]
            if md is None or not _is_bound_cache(kv_cache):
                comp = self._dsv4_compress(hidden_states)  # [Ncomp,Hd] VALUES or None
                o_w, lse_w = self._windowed_branch(
                    q.view(1, T, N, Hd), kv.view(1, T, Hd)
                )
                if comp is None:
                    # sub-window chunk: window+sink only (no compressed slots yet)
                    o_c, lse_c = None, None
                else:
                    topk = self._dsv4_indexer(qr, hidden_states, positions)  # [T,k]
                    o_c, lse_c = self._sparse_compressed_branch(
                        q.view(1, T, N, Hd), comp.unsqueeze(0), topk.unsqueeze(0)
                    )
                o = self._two_branch_merge(o_w, lse_w, o_c, lse_c, sink)  # [1,N,T,Hd]
                return o.permute(0, 2, 1, 3).reshape(T, N, Hd)
            users = md.cache_position.shape[0]
            assert T % users == 0, f"tokens ({T}) not divisible by users ({users})"
            S = T // users
            q4 = q.view(users, S, N, Hd)
            kv3 = kv.view(users, S, Hd)
            if S > 1:
                o = self._c4a_paged_prefill(
                    q4, kv3, hidden_states, positions, qr, sink, kv_cache, md
                )
            else:
                o = self._c4a_paged_decode(
                    q4, kv3, hidden_states, positions, qr, sink, kv_cache, md
                )
            return o.reshape(T, N, Hd)

        def _indexer_key_cache_and_metadata(self):
            """(indexer compressed-KEYS cache, TTMetadata) for the C4A keys group.
            The keys share the full-MLA group's block table with the compressed
            VALUES, so the metadata is the same compressed TTMetadata (keyed by the
            indexer k_cache prefix in the per-layer fan-out)."""
            kc = getattr(getattr(self, "indexer", None), "k_cache", None)
            key_cache = getattr(kc, "kv_cache", None) if kc is not None else None
            md = None
            try:
                from vllm.forward_context import get_forward_context

                fc = get_forward_context()
                attn_md = getattr(fc, "attn_metadata", None)
                if isinstance(attn_md, dict) and kc is not None:
                    md = attn_md.get(kc.prefix)
                elif not isinstance(attn_md, dict):
                    md = attn_md
            except Exception:
                md = None
            return key_cache, md

        def _compressor_state_buf(self, comp):
            """The bound rolling-state buffer for a DeepseekCompressor, or None.
            Allocated + bound in model_runner.initialize_kv_cache onto the
            compressor's CompressorStateCache module as ``.state_cache.kv_cache``
            (dense fp32 [max_num_reqs, coff*ratio, 2*coff*head_dim]; last dim packs
            [kv_state | score_state], each coff*head_dim wide)."""
            sc = getattr(comp, "state_cache", None)
            buf = getattr(sc, "kv_cache", None) if sc is not None else None
            return buf if _is_bound_cache(buf) else None

        def _seed_compressor_state(self, hs, comp, head_dim, seq_len):
            """Seed a compressor's rolling state at prefill so incremental decode
            continues exactly (oracle Compressor.forward start_pos==0 tail,
            modified_model/model.py:380-400). ``hs`` [users, S, hidden] is the padded
            prefill buffer; ``seq_len`` [users] is the REAL token count P per user
            (= md.cache_position + 1). CRITICAL: seed from the last real tokens by
            index masked against P, NOT the padded buffer length S — otherwise the
            padding tokens [P:S) get seeded as the trailing block and decode continues
            from garbage. For a fresh contiguous prefill token index == position.

            Each token at (real) index i is placed by the SAME rule the decode uses:
            it belongs to the in-progress 'current' block iff ``i//ratio == P//ratio``
            (row ``ratio + i%ratio`` for overlap; ``i%ratio`` for non-overlap), or the
            'previous' complete block iff ``i//ratio == P//ratio - 1`` (row ``i%ratio``,
            overlap only). Everything else (incl. padding i>=P) is dropped; unfilled
            rows stay [kv=0 | score=_MASK_NEG]. Chunked / prefix-cached prefill (where
            position != index) is a follow-up."""
            buf = self._compressor_state_buf(comp)
            if buf is None:
                return
            ratio, coff, overlap = comp.compress_ratio, comp.coff, comp.overlap
            w = coff * head_dim
            users, S, _ = hs.shape
            B, dev = buf.shape[0], hs.device
            W_rows = coff * ratio  # 2*ratio overlap, ratio non-overlap
            # per-token raw projections + per-within-block-position ape on score.
            kvsc = comp.fused_wkv_wgate(hs.reshape(users * S, -1))
            if isinstance(kvsc, (tuple, list)):
                kvsc = kvsc[0]
            kvsc = kvsc.view(users, S, 2 * w).float()
            kv_all, sc_all = kvsc.split([w, w], dim=-1)
            idx = torch.arange(S, device=dev)  # token index == position (fresh prefill)
            sc_all = sc_all + comp.ape.float().index_select(0, idx % ratio)  # [u,S,w]
            tok = torch.cat([kv_all, sc_all], dim=-1)  # [users, S, 2w]
            P = seq_len.to(dev).view(users, 1)  # [users, 1] real length
            cur_blk = P // ratio  # [users, 1] in-progress block at position P
            blk = (idx // ratio).view(1, S)  # [1, S]
            within = (idx % ratio).view(1, S)  # [1, S]
            real = idx.view(1, S) < P  # [users, S]
            in_cur = (blk == cur_blk) & real
            neg1 = torch.full((users, S), -1, device=dev, dtype=torch.long)
            if overlap:
                in_prev = (blk == (cur_blk - 1)) & real  # complete block
                row = torch.where(
                    in_cur,
                    ratio + within,
                    torch.where(in_prev, within.expand(users, S), neg1),
                )  # [users, S]
            else:
                row = torch.where(in_cur, within.expand(users, S), neg1)
            # scatter each token into state[row] via a one-hot bmm (each row gets one
            # token; rows with none keep the init fill). Avoids the 5D-broadcast trap.
            onehot = (row.unsqueeze(-1) == torch.arange(W_rows, device=dev)).to(
                tok.dtype
            )  # [users, S, W_rows]
            oh_t = onehot.transpose(1, 2)  # [users, W_rows, S]
            filled = torch.bmm(oh_t, tok)  # [users, W_rows, 2w]
            # per-row token count via a matmul (avoid a bool `any`/reduce that
            # tt-mlir won't legalize); >0 marks a row that received a token.
            cnt = torch.bmm(oh_t, tok.new_ones(users, S, 1)).squeeze(-1)  # [users,W]
            has = cnt > 0.5  # [users, W_rows]
            init = torch.cat(
                [torch.zeros(w, device=dev), torch.full((w,), _MASK_NEG, device=dev)]
            )  # [2w]
            state = torch.where(has.unsqueeze(-1), filled, init.view(1, 1, 2 * w))
            state = state.to(buf.dtype)
            new_buf = state if users == B else torch.cat([state, buf[users:]], dim=0)
            buf.copy_(new_buf)

        def _compress_decode_step(self, hs, positions, comp, head_dim, rope_cache):
            """One incremental compressor step for a single decode token per user
            (oracle Compressor.forward start_pos>0, modified_model/model.py:405-449).
            Handles BOTH the overlap compressor (C4A, coff==2) and the non-overlap
            compressor (C128A, coff==1). Updates the rolling state IN PLACE and
            returns (slot [users, head_dim], write_slot [users] int32 = pos//ratio).
            ``hs`` [users, hidden], ``positions`` [users]. The returned slot is the
            candidate compressed latent for slot ``write_slot``; it is EXACT on a
            boundary step ((pos+1)%ratio==0) and a harmless partial on other steps
            (the caller writes it to the in-progress slot, which the reader excludes
            via comp cache_position=(pos+1)//ratio-1)."""
            buf = self._compressor_state_buf(comp)
            ratio, coff, overlap = comp.compress_ratio, comp.coff, comp.overlap
            d, rope = head_dim, self.rope_head_dim
            nope, w = d - rope, coff * head_dim
            W_rows = coff * ratio  # state rows: 2*ratio overlap, ratio non-overlap
            users = hs.shape[0]
            B, dev = buf.shape[0], hs.device
            assert users == B, f"decode users ({users}) != state rows ({B})"
            # this token's raw projection + per-within-block-position ape on score
            kvsc = comp.fused_wkv_wgate(hs)
            if isinstance(kvsc, (tuple, list)):
                kvsc = kvsc[0]
            kv_t, sc_t = kvsc.float().split([w, w], dim=-1)  # each [users, w]
            j = (positions % ratio).long()  # within-block index [users]
            sc_t = sc_t + comp.ape.float().index_select(0, j)  # [users, w]
            tok = torch.cat([kv_t, sc_t], dim=-1)  # [users, 2w]
            # functional scatter into the state row via one-hot where. Overlap writes
            # the CURRENT-block half (rows [ratio, 2ratio)); non-overlap writes the
            # single ratio-row ring at index pos%ratio.
            row = (ratio + j) if overlap else j  # [users]
            row_oh = torch.arange(W_rows, device=dev).view(1, -1) == row.view(-1, 1)
            state = torch.where(
                row_oh[:, :, None], tok[:, None, :], buf.float()
            )  # [users, W_rows, 2w]
            kv_s, sc_s = state[..., :w], state[..., w:]
            if overlap:
                # overlap-combine the 2*ratio window: prev-block first-half channels
                # ++ current-block second-half channels, per-channel softmax pool.
                kv_win = torch.cat([kv_s[:, :ratio, :d], kv_s[:, ratio:, d:]], dim=1)
                sc_win = torch.cat([sc_s[:, :ratio, :d], sc_s[:, ratio:, d:]], dim=1)
            else:
                # non-overlap (w == d): pool all ratio rows of the current block.
                kv_win, sc_win = kv_s, sc_s
            slot = (kv_win * sc_win.softmax(dim=1)).sum(dim=1)  # [users, d]
            slot = comp.norm(slot.to(hs.dtype))  # RMSNorm(d)
            write_slot = (positions // ratio).long()  # [users]
            cos_c, sin_c = _dsv4_cos_sin(rope_cache, write_slot * ratio, rope)
            slot = _dsv4_rope_rope_dims(slot, cos_c, sin_c, nope, headed=False)
            if overlap:
                # roll ONLY where a slot completed: prev block <- current block.
                should = ((positions + 1) % ratio == 0).view(users, 1, 1)
                new_prev = torch.where(should, state[:, ratio:], state[:, :ratio])
                new_state = torch.cat([new_prev, state[:, ratio:]], dim=1)
            else:
                new_state = state  # non-overlap: single ring, no roll
            buf.copy_(new_state.to(buf.dtype))
            return slot, write_slot.to(torch.int32)

        def _c4a_paged_prefill(
            self, q4, kv3, hidden_states, positions, qr, sink, kv_cache, md
        ):
            """Paged C4A prefill: window self-attend + lightning-indexer top-k
            SPARSE compressed attend, merge; then write the SWA, compressed-VALUES,
            and indexer-KEYS caches (for decode). Returns o [users, S, N, Hd]."""
            users, S, N, Hd = q4.shape
            hs = hidden_states.view(users, S, -1)
            pos2d = positions.view(users, S)
            qr2 = qr.view(users, S, -1)
            o_w, lse_w = self._windowed_branch(q4, kv3)
            self._write_swa_cache_prefill(kv3, kv_cache, md)
            # Seed BOTH compressors' rolling state (main + indexer) for exact-across-
            # boundary decode. MUST run even for a sub-window prefill (S < ratio) so
            # the prefill tokens live in the rolling state and the first slot
            # completes correctly at decode.
            self._seed_compressor_state(
                hs, self.compressor, self.head_dim, md.cache_position + 1
            )
            ix = getattr(self, "indexer", None)
            if ix is not None and getattr(ix, "compressor", None) is not None:
                self._seed_compressor_state(
                    hs, ix.compressor, ix.head_dim, md.cache_position + 1
                )
            # Sub-window prefill chunk (S < ratio): no compressed slots complete ->
            # window+sink only (`S` is a trace-time constant; prunes the compressed
            # subgraph for the small compile buckets).
            if S // self.compress_ratio == 0:
                o = self._two_branch_merge(o_w, lse_w, None, None, sink)
                return o.permute(0, 2, 1, 3)
            comp = torch.stack(
                [self._dsv4_compress(hs[b]) for b in range(users)], dim=0
            )  # [users, Ncomp, Hd] main compressed VALUES
            topk = torch.stack(
                [self._dsv4_indexer(qr2[b], hs[b], pos2d[b]) for b in range(users)],
                dim=0,
            )  # [users, S, k]
            o_c, lse_c = self._sparse_compressed_branch(q4, comp, topk)
            o = self._two_branch_merge(o_w, lse_w, o_c, lse_c, sink)  # [users,N,S,Hd]
            comp_cache, comp_md = self._compressed_cache_and_metadata()
            if _is_bound_cache(comp_cache) and comp_md is not None:
                self._write_compressed_cache_prefill(comp, comp_cache, comp_md)
            key_cache, key_md = self._indexer_key_cache_and_metadata()
            if _is_bound_cache(key_cache) and key_md is not None:
                keyc = torch.stack(
                    [self._dsv4_index_compress(hs[b]) for b in range(users)], dim=0
                )  # [users, Ncomp, index_head_dim]
                self._write_compressed_cache_prefill(keyc, key_cache, key_md)
            return o.permute(0, 2, 1, 3)  # [users, S, N, Hd]

        def _c4a_paged_decode(
            self, q4, kv3, hidden_states, positions, qr, sink, kv_cache, md
        ):
            """Paged C4A decode — EXACT ACROSS COMPRESSION BOUNDARIES. The main
            compressor's incremental slot formation + compressed-VALUES write + dense
            read is shared with C128A in :meth:`_c128a_paged_decode` (called below
            with hidden_states/positions). Here we additionally keep the INDEXER's own
            compressed KEYS rolling state + cache slot in lock-step (used by the >2048
            decode-time top-k extension; below 2048 the dense read equals the sparse
            top-k selection so decode is already exact).

            NOTE >2048: the decode-time lightning indexer + sparse gather (genuine
            top-k pruning for context > 2048) is the remaining extension; only the
            decode-time scoring/top-k/sparse-gather is left since the KEYS state is
            maintained here."""
            users = q4.shape[0]
            comp_cache, comp_md = self._compressed_cache_and_metadata()
            pos = positions.view(users)
            # indexer KEYS rolling state + cache slot (C4A only)
            ix = getattr(self, "indexer", None)
            key_cache, key_md = self._indexer_key_cache_and_metadata()
            if (
                ix is not None
                and getattr(ix, "compressor", None) is not None
                and self._compressor_state_buf(ix.compressor) is not None
                and _is_bound_cache(key_cache)
                and key_md is not None
            ):
                kslot, kws = self._compress_decode_step(
                    hidden_states.view(users, -1),
                    pos,
                    ix.compressor,
                    ix.head_dim,
                    self.indexer_rotary_emb.cos_sin_cache,
                )
                kfill = kslot.unsqueeze(1).transpose(0, 1).unsqueeze(2)
                key_cache.copy_(
                    torch.ops.tt.paged_update_cache(
                        key_cache,
                        kfill.to(key_cache.dtype),
                        kws,
                        key_md.page_table,
                    )
                )
            # main-compressor incremental write + dense read (shared with C128A;
            # exact for context <= 2048 across boundaries).
            return self._c128a_paged_decode(
                q4,
                kv3,
                sink,
                kv_cache,
                md,
                comp_cache,
                comp_md,
                hidden_states,
                positions,
            )


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
