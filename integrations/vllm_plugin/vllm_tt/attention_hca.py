# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""
HCA (High-Compression Attention) backend for TT devices — DeepSeek v4.

DeepSeek v4 splits its attention layers into three types, keyed by the per-layer
``compress_ratio`` (``config.compress_ratios[layer_id]``):

* ``1``   — sliding-window-only (SWA), no KV compression.
* ``4``   — C4A: SWA + a learned sparse indexer (top-k) + 4:1 KV compression.
* ``128`` — **C128A: SWA + 128:1 KV compression, no indexer.**

HCA is the C128A path — the *high-compression* layers. This module ports that
attention to TT and mirrors vLLM 0.20.1's
``vllm.model_executor.layers.deepseek_v4_attention.DeepseekV4MLAAttention``
(``forward(q, kv, positions, output)``).

Two code paths exist, chosen per layer by ``compress_ratio``:

* **Single-cache** — non-C128A layers use ``TTHCAAttentionBackendImpl.forward``:
  one bf16 paged latent cache via the TT MLA ops (``flash_mla_prefill`` /
  ``paged_flash_mla_decode``).
* **Dual-cache (C128A) — LIVE (runner-paged)** — ``TTHCAAttention.forward_dual``
  runs the real high-compression attention: a ``TTHCACompressor``
  (softmax-gated pooling → RMSNorm → RoPE) builds the compressed global pool, and
  the decode does a single joint softmax over (compressed pool ∪ SWA window) with
  the attention sink (``forward_dual_decode`` / ``hca_dual_attention``).

  vLLM fuses two FP8 caches inside one FlashMLA kernel; the TT MLA ops attend
  over a single cache only (no LSE / sink-in-prefill), so the combine is explicit
  (gather both caches → one joint softmax). The compressed pool + the compressor
  state caches (``cstate_kv`` / ``cstate_score``) are registered as identical-spec
  ``MLAAttention`` sub-layers (``TTHCAPagedCache``) so the existing model runner
  allocates + binds them with **no runner changes**; they land in the same
  KV-cache group and share group 0's block table, so the single
  ``TTMetadata.page_table`` / ``cache_position`` addresses every cache. Compressed
  length and compression boundaries are derived from ``cache_position``
  (``(pos+1)//ratio``), so generated tokens are compressed incrementally during
  decode via the state caches — no per-step metadata from the runner is needed.

Both paths are bf16 (no FP8 ``head_bytes`` packing); ``head_size`` is the bf16
latent width. The secondary caches are currently full-size (same spec as the
window cache) — right-sizing them (a bounded sliding-window window cache + a
smaller compressed/state group) is a memory optimisation that needs multi-group
runner support; see HCA_CHANGES.md.

Latent layout (the "high compression" part): unlike standard MLA — where the
cache stores ``kv_c (kv_lora_rank)`` concatenated with a *separate* ``k_pe``
(rope) to give ``kv_lora_rank + qk_rope_head_dim`` — DeepSeek v4 fuses the whole
KV stream into a single ``head_dim`` latent with **rope embedded in-place in the
trailing ``qk_rope_head_dim`` dims** (vLLM hardcodes ``kv_lora_rank == head_dim``
and ``v_head_dim == head_dim``). K and V are the same latent (``value=None`` in
the TT ops reuses the leading ``head_dim_v`` features of K as V), so the paged
cache is a single latent head of width ``head_dim``:
``[num_blocks, 1, block_size, head_dim]``.

Both ``q`` and ``kv`` are expected to arrive with rope already applied (the outer
DeepSeek v4 MLA layer does this before attention), exactly as ``q``/``kv`` reach
``DeepseekV4MLAAttention.forward``.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.deepseek_v4_attention import (
    DeepseekV4MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.v1.attention.backend import AttentionBackend, MLAAttentionImpl

from .attention import TTAttentionMetadataBuilder, TTMetadata
from .logger import tt_init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = tt_init_logger(__name__)


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class TTHCAAttentionBackend(AttentionBackend):
    """vLLM attention backend for DeepSeek v4 High-Compression Attention on TT."""

    @staticmethod
    def get_name() -> str:
        return "TT_HCA"

    @staticmethod
    def get_impl_cls() -> type["TTHCAAttentionBackendImpl"]:
        return TTHCAAttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        # Reuse the stub builder shared by the other TT backends; HCA consumes
        # the same TTMetadata the model runner constructs.
        return TTAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Single fused latent KV head per slot (K and V share one compressed
        # latent), matching DeepseekV4's MLAAttentionSpec(num_kv_heads=1,
        # head_size=head_dim). `head_size` here is the full head_dim (rope is
        # embedded in its trailing qk_rope_head_dim dims).
        assert num_kv_heads == 1, "num_kv_heads must be 1 for HCA"
        return (num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_page_size(vllm_config: "VllmConfig") -> int:
        return 32

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TT HCA backend.")


# --------------------------------------------------------------------------- #
# Impl
# --------------------------------------------------------------------------- #
class TTHCAAttentionBackendImpl(MLAAttentionImpl):
    """
    High-Compression Attention impl for TT (DeepSeek v4 C128A).

    Mirrors ``DeepseekV4MLAAttention.forward(q, kv, positions, output)``: the
    query is the full per-head latent and ``kv`` is the single fused latent K/V
    (both rope-embedded). ``forward_mha`` / ``forward_mqa`` are never called.
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
        # DeepSeek v4 HCA-specific arguments
        window_size: Optional[int] = None,
        compress_ratio: Optional[int] = None,
        attn_sink: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        # The fused latent width == DeepseekV4's head_dim (rope embedded in the
        # trailing qk_rope_head_dim dims). vLLM passes this as head_size and also
        # sets kv_lora_rank == v_head_dim == head_dim.
        self.head_dim = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        # V is read straight from the latent K; in DeepseekV4 v_head_dim == head_dim.
        self.v_head_dim = v_head_dim if v_head_dim else head_size

        # SWA window + compression ratio (carried for the future SWA/compressed
        # split; the current TT path attends over the full causal latent cache).
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        # Per-head attention sink logits (DeepseekV4 ``attn_sink``); applied in
        # decode only (the TT prefill op does not support sinks).
        self.attn_sink = attn_sink

        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes are not supported for HCA on TT.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                f"Quantized HCA KV cache ({kv_cache_dtype}) is not yet "
                "supported on TT (DeepseekV4 uses FP8; TT bring-up is bf16)."
            )

        # The latent is one fused stream: nope dims followed by rope dims.
        assert head_size == qk_nope_head_dim + qk_rope_head_dim, (
            f"HCA latent width head_size ({head_size}) must equal "
            f"qk_nope_head_dim ({qk_nope_head_dim}) + qk_rope_head_dim "
            f"({qk_rope_head_dim})."
        )
        # V is taken from the latent, so its head dim cannot exceed the latent.
        assert 0 < self.v_head_dim <= head_size, (
            f"HCA v_head_dim ({self.v_head_dim}) must be in (0, head_dim "
            f"= {head_size}]."
        )

    # ------------------------------------------------------------------ #
    # Abstract stubs — never called; the TT MLA layer routes through
    # forward() directly.
    # ------------------------------------------------------------------ #
    def forward_mha(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "TTHCAAttentionBackendImpl.forward_mha should never be called; "
            "the TT HCA layer routes through forward() directly."
        )

    def forward_mqa(self, *args, **kwargs):
        raise RuntimeError(
            "TTHCAAttentionBackendImpl.forward_mqa should never be called; "
            "the TT HCA layer routes through forward() directly."
        )

    # ------------------------------------------------------------------ #
    # Unified forward — handles both prefill and decode.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """High-Compression Attention on TT (prefill and paged decode).

        Dispatches on token count per user: prefill (S > 1) attends against the
        freshly built local latent K via ``tt.flash_mla_prefill``; decode
        (S == 1) attends against the paged latent KV cache via
        ``tt.paged_flash_mla_decode``.

        Shapes (matching ``DeepseekV4MLAAttention.forward``; rope already
        applied by the outer layer):
            q:        [tokens, num_heads, head_dim]
            kv:       [tokens, head_dim]                fused latent K (== V)
            kv_cache: [num_blocks, 1, block_size, head_dim]
            output:   [tokens, num_heads * v_head_dim]  (optional write target)
        Returns the written output tensor.
        """
        is_prefill = self._infer_is_prefill(q, attn_metadata)

        users = (
            attn_metadata.cache_position.shape[0]
            if attn_metadata is not None and attn_metadata.cache_position is not None
            else 1
        )
        total_tokens = q.shape[0]
        assert (
            total_tokens % users == 0
        ), f"total_tokens ({total_tokens}) not divisible by users ({users})."
        S = total_tokens // users
        N = self.num_heads
        D = self.head_dim

        act_dtype = q.dtype

        # -- Reshape to [users, S, ...]. q is the full latent query; kv is the
        #    single fused latent K/V head. No W_UK_T / W_UV (non-absorbed). -----
        q_lat = q.view(users, S, N, D)
        k_lat = kv.view(users, S, 1, D)

        if is_prefill:
            return self._forward_prefill(
                q_lat, k_lat, kv_cache, attn_metadata, S, users, act_dtype, output
            )
        else:
            return self._forward_decode(
                q_lat, k_lat, kv_cache, attn_metadata, users, act_dtype, output
            )

    @staticmethod
    def _infer_is_prefill(q: torch.Tensor, attn_metadata: TTMetadata | None) -> bool:
        """
        Prefill when more than one token per user, decode otherwise.
        The scheduler guarantees that a batch handed to this class is either all
        prefill requests or all decode requests.
        """
        if attn_metadata is None or attn_metadata.cache_position is None:
            # Treat profiling runs as prefill.
            return True
        users = attn_metadata.cache_position.shape[0]
        assert users > 0, "Invalid number of users"
        total_tokens = q.shape[0]
        return (total_tokens // users) > 1

    def _forward_prefill(
        self,
        q_lat: torch.Tensor,
        k_lat: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        seq_len: int,
        users: int,
        act_dtype: torch.dtype,
        output: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q_for_kernel = q_lat.transpose(1, 2).contiguous()  # [b, N, S, D]
        k_for_kernel = k_lat.transpose(1, 2).contiguous()  # [b, 1, S, D]

        # value=None: V is read from the first v_head_dim features of the latent
        # K (K and V share the compressed latent). No W_UV expansion follows.
        out = torch.ops.tt.flash_mla_prefill(
            query=q_for_kernel,
            key=k_for_kernel,
            head_dim_v=self.v_head_dim,
            value=None,
            is_causal=attn_metadata.is_causal if attn_metadata is not None else True,
            scale=self.scale,
        ).to(
            act_dtype
        )  # [b, N, S, v_head_dim]

        # Reshape to vLLM's output contract: [tokens, N * v_head_dim].
        out = out.transpose(1, 2).reshape(
            users * seq_len, self.num_heads * self.v_head_dim
        )

        # Persist tokens in the latent KV cache (skipped during profiling runs).
        if (
            attn_metadata is not None
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.numel() > 0
        ):
            k_lat_for_fill = k_lat.transpose(1, 2)  # [b, 1, S, D]
            fill_page_table = attn_metadata.fill_page_table
            # Accumulate per-user fills into a local so kv_cache keeps
            # referencing the bound buffer (the loop must not rebind it).
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

        if output is not None:
            output.copy_(out)
            return output
        return out

    def _forward_decode(
        self,
        q_lat: torch.Tensor,
        k_lat: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        users: int,
        act_dtype: torch.dtype,
        output: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Paged HCA decode on TT (one token per user, S == 1).
        Shapes:
            q_lat:    [users, 1, N, D]                  latent query (S == 1)
            k_lat:    [users, 1, 1, D]                  new token's latent K
            kv_cache: [num_blocks, 1, block_size, D]    paged latent cache
        """
        # Write the new token's latent K into the paged cache at the current
        # position (skipped during profiling runs).
        if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() > 0:
            k_lat_for_update = k_lat.transpose(0, 1)  # [1, users, 1, D]
            updated_cache = torch.ops.tt.paged_update_cache(
                kv_cache,
                k_lat_for_update,
                attn_metadata.cache_position,
                attn_metadata.page_table,
            )
            kv_cache.copy_(updated_cache)

        # Paged MLA decode kernel: query [1, users, N, D], reads K (and V, via
        # value=None) straight from the paged latent cache. attn_sink mirrors
        # DeepseekV4's per-head sink logits.
        is_causal = attn_metadata.is_causal if attn_metadata is not None else True
        out = torch.ops.tt.paged_flash_mla_decode(
            query=q_lat.transpose(0, 1),  # [1, users, N, D]
            key=kv_cache,
            head_dim_v=self.v_head_dim,
            page_table=attn_metadata.page_table,
            value=None,
            is_causal=is_causal,
            attn_mask=None if is_causal else attn_metadata.attn_mask,
            cur_pos_tensor=attn_metadata.cache_position,
            attention_sink=self.attn_sink,
            scale=self.scale,
        ).to(
            act_dtype
        )  # [1, users, N, v_head_dim]

        # Reshape to vLLM's output contract: [tokens, N * v_head_dim].
        out = out.reshape(users, self.num_heads * self.v_head_dim)
        if output is not None:
            output.copy_(out)
            return output
        return out

    # ------------------------------------------------------------------ #
    # Dual-cache decode (SWA window + compressed pool) — DeepSeek v4 C128A.
    #
    # The TT MLA ops attend over a single cache only, so the two-cache joint
    # softmax (which vLLM fuses inside one FlashMLA kernel) is done explicitly
    # here: gather both caches to dense, then one joint softmax with the sink
    # (see ``hca_dual_attention``). Cache *writes* still use the paged ops.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gather_latent(cache: torch.Tensor, page_table: torch.Tensor) -> torch.Tensor:
        """Gather a paged single-head latent cache into a dense per-user tensor.

        cache:      [num_blocks, 1, block_size, head_dim]
        page_table: [users, num_blocks_per_user]  (physical block ids, logical order)
        returns:    [users, num_blocks_per_user * block_size, head_dim]
        """
        users, bpu = page_table.shape
        block_size = cache.shape[2]
        head_dim = cache.shape[3]
        flat = page_table.reshape(-1)
        gathered = torch.index_select(cache, 0, flat)  # [users*bpu, 1, block, D]
        gathered = gathered.view(users, bpu, block_size, head_dim)
        return gathered.reshape(users, bpu * block_size, head_dim)

    def forward_dual_decode(
        self,
        q: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_page_table: torch.Tensor,
        comp_cache: torch.Tensor,
        comp_page_table: torch.Tensor,
        cur_pos: torch.Tensor,
        num_compressed: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One decode step of dual-cache HCA (one token per user).

        Attends jointly over the sliding window (last ``window_size`` tokens of
        ``swa_cache``) and the full compressed pool (first ``num_compressed``
        tokens of ``comp_cache``), with the per-head attention sink.

        q:               [users, num_heads, head_dim]   (rope already applied)
        swa_cache:       [num_blocks, 1, swa_block, head_dim]
        comp_cache:      [num_blocks, 1, comp_block, head_dim]
        cur_pos:         [users]  current absolute position per user (for the window)
        num_compressed:  [users]  number of valid compressed tokens per user
        returns:         [users, num_heads * v_head_dim]
        """
        users = q.shape[0]
        device = q.device

        k_swa = self._gather_latent(swa_cache, swa_page_table)  # [u, Tw, D]
        k_comp = self._gather_latent(comp_cache, comp_page_table)  # [u, Tc, D]
        Tw = k_swa.shape[1]
        Tc = k_comp.shape[1]

        # SWA mask: valid sliding window = positions (cur_pos - window, cur_pos].
        win = self.window_size if self.window_size is not None else Tw
        wpos = torch.arange(Tw, device=device).unsqueeze(0)  # [1, Tw]
        cur = cur_pos.view(users, 1)
        swa_mask = (wpos <= cur) & (wpos > cur - win)  # [u, Tw]
        # Compressed mask: first num_compressed entries valid.
        cpos = torch.arange(Tc, device=device).unsqueeze(0)  # [1, Tc]
        comp_mask = cpos < num_compressed.view(users, 1)  # [u, Tc]

        out = hca_dual_attention(
            q,
            [k_comp, k_swa],
            self.attn_sink,
            self.scale,
            self.v_head_dim,
            masks=[comp_mask, swa_mask],
        )  # [u, N, v_head_dim]

        out = out.reshape(users, self.num_heads * self.v_head_dim)
        if output is not None:
            output.copy_(out)
            return output
        return out


# --------------------------------------------------------------------------- #
# Dual-cache helpers (SWA window + compressed pool) — DeepSeek v4 C128A
# --------------------------------------------------------------------------- #
def _hca_gated_compress(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, int]:
    """Softmax-gated pooling of ``compress_ratio`` consecutive tokens into one.

    Implements DeepSeek v4's C128A (non-overlapping) compression:
        score_w = score + ape   (ape is the per-position-in-window additive PE)
        w       = softmax(score_w, over the window)
        out     = sum_window(kv * w)

    kv, score: [num_tokens, head_dim]; ape: [compress_ratio, head_dim].
    Only full windows are pooled; the trailing ``num_tokens % compress_ratio``
    tokens are returned as the leftover count (they belong in the SWA window, not
    the compressed pool). Assumes tokens are contiguous from a window boundary so
    the i-th token in a window gets ``ape[i]``.

    Returns (compressed [num_windows, head_dim], leftover_token_count).
    """
    total_tokens, head_dim = kv.shape
    num_full = total_tokens // compress_ratio
    if num_full == 0:
        return kv.new_zeros((0, head_dim)), total_tokens
    n = num_full * compress_ratio
    kv_w = kv[:n].view(num_full, compress_ratio, head_dim)
    score_w = score[:n].view(num_full, compress_ratio, head_dim) + ape.unsqueeze(0)
    w = torch.softmax(score_w, dim=1)
    return (kv_w * w).sum(dim=1), total_tokens - n


def _masked_gated_pool(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Softmax-gated pool over a per-user masked window (decode path).

    Pools the compressor states of the tokens in each user's *current* window
    into one latent: ``score += ape[pos % ratio]``, softmax over the masked
    window, weighted sum of ``kv``. Tokens outside the window (``mask == False``)
    get -inf logits and drop out.

    kv, score: [users, T, head_dim]   (per-token compressor states)
    ape:       [compress_ratio, head_dim]
    positions: [users, T]             absolute position of each cached token
    mask:      [users, T]             True for tokens in the current window
    returns:   [users, head_dim]
    """
    score = score + ape[(positions % compress_ratio)]
    score = score.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    w = torch.softmax(score, dim=1)
    return (kv * w).sum(dim=1)


def hca_dual_attention(
    q: torch.Tensor,
    k_blocks: list[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    scale: float,
    v_head_dim: int,
    masks: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """Joint softmax attention over the union of latent KV blocks + a sink.

    This is the single-softmax combine DeepSeek v4 performs over the compressed
    pool ∪ the SWA window (MLA-from-latent: V is the leading ``v_head_dim``
    features of each latent K). One query token per user (decode).

    q:         [users, num_heads, head_dim]
    k_blocks:  list of [users, n_i, head_dim] (concatenated along the key axis)
    masks:     optional list of [users, n_i] bool (True = valid); padded/invalid
               keys get -inf logits.
    attn_sink: [num_heads] per-head sink logits (added as one extra softmax
               column, then dropped), or None.
    returns:   [users, num_heads, v_head_dim]
    """
    k = torch.cat(k_blocks, dim=1)  # [u, K, D]
    scores = torch.einsum("unh,ukh->unk", q, k) * scale  # [u, N, K]
    if masks is not None:
        m = torch.cat(masks, dim=1)  # [u, K]
        scores = scores.masked_fill(~m.unsqueeze(1), float("-inf"))
    if attn_sink is not None:
        u, num_heads, _ = scores.shape
        sink_col = (
            attn_sink.view(1, num_heads, 1).to(scores.dtype).expand(u, num_heads, 1)
        )
        logits = torch.cat([scores, sink_col], dim=-1)  # [u, N, K+1]
        weights = torch.softmax(logits, dim=-1)[..., :-1]  # drop the sink column
    else:
        weights = torch.softmax(scores, dim=-1)
    v = k[..., :v_head_dim]  # [u, K, v_head_dim]
    return torch.einsum("unk,ukh->unh", weights, v)  # [u, N, v_head_dim]


class TTHCACompressor(nn.Module):
    """Clean bf16 DeepSeek v4 KV compressor (C128A, non-overlapping).

    Mirrors vLLM's ``DeepseekCompressor`` module surface (``fused_wkv_wgate``,
    ``ape``, ``norm``) so checkpoint weights load, but runs a readable torch
    compression instead of the fused Triton/FP8 kernels: softmax-gated pooling of
    ``compress_ratio`` tokens → RMSNorm → RoPE (at the window-start position).
    """

    def __init__(
        self,
        *,
        vllm_config: "VllmConfig",
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        prefix: str,
    ) -> None:
        super().__init__()
        # C4A's overlapping (coff=2) compressor is not supported yet — only the
        # high-compression C128A path.
        assert compress_ratio != 4, (
            "C4A (overlapping) compressor is not supported on TT yet; "
            "TTHCACompressor handles the C128A non-overlapping path."
        )
        config = vllm_config.model_config.hf_config
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            hidden_size,
            [head_dim, head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(head_dim, self.rms_norm_eps)
        self.ape = nn.Parameter(
            torch.zeros(compress_ratio, head_dim), requires_grad=False
        )

    def project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token compressor projection (no pooling).

        hidden_states: [num_tokens, hidden_size]
        Returns (kv [num_tokens, head_dim], score [num_tokens, head_dim]) — the
        raw gate score WITHOUT ``ape`` (the additive PE is folded in at pool time,
        keyed by each token's position-in-window, by the pool helpers).
        """
        kv_score = self.fused_wkv_wgate(hidden_states)
        return kv_score.split([self.head_dim, self.head_dim], dim=-1)

    def norm_rope(
        self,
        pooled: torch.Tensor,
        comp_positions: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        """RMSNorm + RoPE a pooled compressed latent at its window-start position.

        pooled:        [..., head_dim]
        comp_positions:[...]  the window-start position for each pooled token.
        """
        pooled = self.norm(pooled)
        out, _ = rotary_emb.forward_native(
            comp_positions.reshape(-1), pooled.reshape(-1, 1, self.head_dim), None
        )
        return out.reshape(pooled.shape)

    def compress(
        self,
        hidden_states: torch.Tensor,
        rotary_emb,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress a contiguous prompt (from a window boundary) into latents.

        hidden_states: [num_tokens, hidden_size]
        Returns (compressed [num_windows, head_dim], comp_positions [num_windows]).
        Only full windows are produced; trailing tokens stay in the SWA window.
        """
        kv, score = self.project(hidden_states)
        compressed, _leftover = _hca_gated_compress(
            kv, score, self.ape, self.compress_ratio
        )
        num_windows = compressed.shape[0]
        if num_windows == 0:
            return compressed, hidden_states.new_zeros((0,), dtype=torch.long)
        # Compressed token i summarises tokens [i*ratio, (i+1)*ratio); its rope
        # position is the window-start i*ratio (== ((i+1)*ratio-1)//ratio*ratio).
        comp_positions = (
            torch.arange(num_windows, device=hidden_states.device) * self.compress_ratio
        )
        compressed = self.norm_rope(compressed, comp_positions, rotary_emb)
        return compressed, comp_positions


# --------------------------------------------------------------------------- #
# Minimal paged-cache sub-layer (compressed pool + compressor state)
# --------------------------------------------------------------------------- #
class TTHCAPagedCache(MLAAttention):
    """A bare single-latent paged cache, owned by an HCA layer.

    Exists only so the TT model runner allocates + binds a cache for it
    (recognised via ``isinstance(module, MLAAttention)`` in the runner's
    ``get_kv_cache_spec``). Its ``MLAAttentionSpec`` is identical to the window
    cache's (``num_kv_heads=1``, ``head_size=head_dim``), so it lands in the same
    KV-cache group and **shares group 0's block table** — letting the HCA layer
    reuse the single ``TTMetadata.page_table`` for every cache. Never attends;
    ``forward`` must not be called.
    """

    def __init__(
        self, *, vllm_config: "VllmConfig", head_dim: int, prefix: str
    ) -> None:
        nn.Module.__init__(self)
        self.num_heads = 1
        self.scale = 1.0
        self.qk_nope_head_dim = head_dim
        self.qk_rope_head_dim = 0
        self.v_head_dim = head_dim
        self.kv_lora_rank = head_dim
        self.head_size = head_dim
        self.num_kv_heads = 1
        self.qk_head_dim = head_dim
        self.layer_name = prefix
        self.kv_cache_dtype = "auto"
        self.kv_b_proj = None
        self.indexer = None
        self.kv_cache = torch.tensor([])
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        pass

    def forward(self, *args, **kwargs):
        raise RuntimeError("TTHCAPagedCache.forward should never be called.")


# --------------------------------------------------------------------------- #
# Inner attention sub-layer (owns the paged latent KV cache(s))
# --------------------------------------------------------------------------- #
class TTHCAAttention(MLAAttention):
    """Inner HCA attention sub-layer for DeepSeek v4.

    Subclasses ``MLAAttention`` purely so the TT model runner recognises it
    (``isinstance(module, MLAAttention)`` in ``get_kv_cache_spec``) and
    allocates a single latent MLA cache — ``[num_blocks, 1, block_size,
    head_dim]`` — then binds it to ``self.kv_cache`` via ``bind_kv_cache``. It
    does **not** use ``MLAAttention``'s standard absorbed machinery (there is no
    ``kv_b_proj`` in DeepSeek v4), so ``__init__`` is custom.

    For high-compression (C128A) layers it also owns three sibling paged caches
    (``compressed`` pool + ``cstate_kv`` / ``cstate_score`` compressor states),
    all the same spec so they share group 0's block table, and runs the
    runner-paged dual cache via ``forward_dual``. Non-C128A layers use the
    single-cache ``forward``.
    """

    def __init__(
        self,
        *,
        vllm_config: "VllmConfig",
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        window_size: Optional[int],
        compress_ratio: Optional[int],
        attn_sink: Optional[torch.Tensor],
        prefix: str,
    ) -> None:
        # Bypass MLAAttention.__init__ (it requires kv_b_proj and routes the
        # backend via get_attn_backend); initialise the bare nn.Module instead.
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # DeepSeek v4 fuses the whole KV stream into one head_dim latent.
        self.kv_lora_rank = head_dim
        self.head_size = head_dim
        self.num_kv_heads = 1
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.layer_name = prefix
        self.kv_cache_dtype = "auto"
        self.kv_b_proj = None
        self.indexer = None
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.head_dim = head_dim

        self.impl = TTHCAAttentionBackendImpl(
            num_heads=num_heads,
            head_size=head_dim,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=window_size,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            window_size=window_size,
            compress_ratio=compress_ratio,
            attn_sink=attn_sink,
        )

        # Placeholder; bound to the (window) latent cache tensor by bind_kv_cache.
        self.kv_cache = torch.tensor([])

        # Register so the model runner can find this layer (build its kv cache
        # spec and bind the allocated cache).
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # C128A: the compressed pool + compressor-state paged caches. Same spec
        # as the window cache -> same group -> shared block table.
        self.dual = self.compress_ratio == 128
        self.compressed = None
        self.cstate_kv = None
        self.cstate_score = None
        if self.dual:
            self.compressed = TTHCAPagedCache(
                vllm_config=vllm_config,
                head_dim=head_dim,
                prefix=f"{prefix}.compressed",
            )
            self.cstate_kv = TTHCAPagedCache(
                vllm_config=vllm_config, head_dim=head_dim, prefix=f"{prefix}.cstate_kv"
            )
            self.cstate_score = TTHCAPagedCache(
                vllm_config=vllm_config,
                head_dim=head_dim,
                prefix=f"{prefix}.cstate_score",
            )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # HCA has no kv_b_proj / W_UK / W_UV to absorb — nothing to do.
        pass

    def _attn_metadata(self) -> "TTMetadata":
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(self.layer_name)
        return attn_metadata

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return self.impl.forward(
            q=q,
            kv=kv,
            kv_cache=self.kv_cache,
            attn_metadata=self._attn_metadata(),
            output=output,
        )

    # ------------------------------------------------------------------ #
    # Runner-paged dual cache (C128A): window + compressed + compressor state.
    # All caches share group 0's block table (attn_metadata.page_table) and
    # cache_position; compressed length + compression boundaries are derived
    # from cache_position, so NO model-runner metadata changes are needed.
    # ------------------------------------------------------------------ #
    def forward_dual(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        hidden_states: torch.Tensor,
        compressor: "TTHCACompressor",
        rotary_emb,
        output: torch.Tensor,
    ) -> torch.Tensor:
        md = self._attn_metadata()
        ratio = self.compress_ratio
        D = self.head_dim
        N = self.num_heads
        page_table = md.page_table
        cache_position = md.cache_position  # [users] absolute pos of last token
        users = cache_position.shape[0]
        total_tokens = q.shape[0]
        S = total_tokens // users

        if S == 1:
            # ---------------- decode ----------------
            device = q.device
            cp = cache_position  # [users]
            # 1) window cache <- main kv (single new token per user)
            k_win = kv.view(users, 1, 1, D).transpose(0, 1)  # [1, users, 1, D]
            self.kv_cache.copy_(
                torch.ops.tt.paged_update_cache(self.kv_cache, k_win, cp, page_table)
            )
            # 2) compressor states <- projected (kv, score) for this token
            kv_c, score_c = compressor.project(hidden_states)  # [users, D] each
            self.cstate_kv.kv_cache.copy_(
                torch.ops.tt.paged_update_cache(
                    self.cstate_kv.kv_cache,
                    kv_c.view(users, 1, 1, D).transpose(0, 1),
                    cp,
                    page_table,
                )
            )
            self.cstate_score.kv_cache.copy_(
                torch.ops.tt.paged_update_cache(
                    self.cstate_score.kv_cache,
                    score_c.view(users, 1, 1, D).transpose(0, 1),
                    cp,
                    page_table,
                )
            )
            # 3) pool the current (partial) window from the state caches and
            #    write it to compressed slot c = pos // ratio. Counting only
            #    completed windows (num_compressed below) ignores partial pools.
            kv_g = self.impl._gather_latent(self.cstate_kv.kv_cache, page_table)
            score_g = self.impl._gather_latent(self.cstate_score.kv_cache, page_table)
            T = kv_g.shape[1]
            tpos = torch.arange(T, device=device).unsqueeze(0).expand(users, T)
            win_start = (cp // ratio * ratio).view(users, 1)
            mask = (tpos >= win_start) & (tpos <= cp.view(users, 1))
            pooled = _masked_gated_pool(
                kv_g, score_g, compressor.ape, tpos, mask, ratio
            )  # [users, D]
            comp_pos = (cp // ratio) * ratio  # window-start position per user
            comp_tok = compressor.norm_rope(pooled, comp_pos, rotary_emb)  # [users, D]
            c_slot = cp // ratio  # [users]
            self.compressed.kv_cache.copy_(
                torch.ops.tt.paged_update_cache(
                    self.compressed.kv_cache,
                    comp_tok.view(users, 1, 1, D).transpose(0, 1),
                    c_slot,
                    page_table,
                )
            )
            num_compressed = (cp + 1) // ratio  # [users] completed windows only
            # 4) joint attention over (windowed window cache) ∪ (compressed pool)
            return self.impl.forward_dual_decode(
                q.view(users, N, D),
                self.kv_cache,
                page_table,
                self.compressed.kv_cache,
                page_table,
                cp,
                num_compressed,
                output,
            )

        # ---------------- prefill ----------------
        # Exact full attention over the window cache (also fills it), then
        # populate the compressor state + compressed caches for later decode.
        out = self.impl.forward(
            q=q,
            kv=kv,
            kv_cache=self.kv_cache,
            attn_metadata=md,
            output=output,
        )
        self._prefill_populate_compression(
            hidden_states, kv, compressor, rotary_emb, md, users, S
        )
        return out

    def _prefill_populate_compression(
        self,
        hidden_states: torch.Tensor,
        kv: torch.Tensor,
        compressor: "TTHCACompressor",
        rotary_emb,
        md: "TTMetadata",
        users: int,
        S: int,
    ) -> None:
        """Write per-token compressor states + the compressed prompt windows.

        Assumes each user's prefill starts at position 0 (no prefix cache); the
        i-th token in a window gets ``ape[i]``. Done per user via paged_fill.
        """
        D = self.head_dim
        ratio = self.compress_ratio
        fill_pt = md.fill_page_table
        kv_c, score_c = compressor.project(hidden_states)  # [users*S, D]
        kv_c = kv_c.view(users, S, D)
        score_c = score_c.view(users, S, D)
        cstate_kv = self.cstate_kv.kv_cache
        cstate_score = self.cstate_score.kv_cache
        compressed = self.compressed.kv_cache
        for u in range(users):
            bidx = torch.tensor([u], dtype=torch.int32, device=kv.device)
            # per-token compressor states (for the trailing partial window that
            # decode will continue from).
            cstate_kv = torch.ops.tt.paged_fill_cache(
                cstate_kv, kv_c[u : u + 1].unsqueeze(1), fill_pt, batch_idx=bidx
            )
            cstate_score = torch.ops.tt.paged_fill_cache(
                cstate_score, score_c[u : u + 1].unsqueeze(1), fill_pt, batch_idx=bidx
            )
            # compress completed windows of this prompt
            comp, comp_pos = compressor.compress(
                hidden_states.view(users, S, -1)[u], rotary_emb
            )
            if comp.shape[0] > 0:
                compressed = torch.ops.tt.paged_fill_cache(
                    compressed,
                    comp.unsqueeze(0).unsqueeze(0),
                    fill_pt,
                    batch_idx=bidx,
                )
        self.cstate_kv.kv_cache.copy_(cstate_kv)
        self.cstate_score.kv_cache.copy_(cstate_score)
        self.compressed.kv_cache.copy_(compressed)


# --------------------------------------------------------------------------- #
# Outer MLA wrapper (OOT replacement for DeepseekV4MultiHeadLatentAttentionWrapper)
# --------------------------------------------------------------------------- #
@DeepseekV4MultiHeadLatentAttentionWrapper.register_oot
class TTDeepseekV4MLAAttention(DeepseekV4MultiHeadLatentAttentionWrapper):
    """TT replacement for DeepSeek v4's outer MLA layer.

    PluggableLayer dispatch (``custom_op.py``) substitutes this class whenever
    the model instantiates ``DeepseekV4MultiHeadLatentAttentionWrapper`` (keyed
    by the base class name). It reuses the projection modules the DeepSeek v4
    model builds (passed via ``mla_modules``) and runs a clean bf16 forward:

        fused_wqa_wkv -> split(qr, kv) -> q_norm / kv_norm -> wq_b -> reshape q
        -> per-head q RMSNorm -> RoPE(q, kv) -> HCA attention (impl)
        -> inverse-RoPE(o) -> grouped wo_a -> wo_b

    The GPU-only machinery of the in-tree wrapper (FP8 quant, CUDA streams,
    fused custom ops, the SWA + compressed dual cache, and the C4A sparse
    indexer) is intentionally omitted: this is the bf16 single-latent-cache HCA
    bring-up. C4A (compress_ratio == 4) layers run the same latent path here,
    minus the indexer (logged once).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        o_lora_rank: Optional[int],
        mla_modules,
        window_size: int,
        compress_ratio: Optional[int],
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        # Skip the in-tree (CUDA/FP8) __init__; build the bare nn.Module.
        nn.Module.__init__(self)
        vllm_config = mla_modules.vllm_config
        config = vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.n_local_groups = config.o_groups // tp_size
        self.eps = config.rms_norm_eps
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1

        # Projection modules built by the DeepSeek v4 model.
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b
        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a
        self.wo_b = mla_modules.wo_b
        self.rotary_emb = mla_modules.rotary_emb

        # Per-head RMSNorm on Q (no learnable weight), as in DeepSeek v4.
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # Inner attention owns the single latent KV cache and runs the impl.
        self.mla_attn = TTHCAAttention(
            vllm_config=vllm_config,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            window_size=window_size,
            compress_ratio=self.compress_ratio,
            attn_sink=mla_modules.attn_sink,
            prefix=prefix,
        )

        # High-compression (C128A) layers own a compressor that builds the
        # compressed global KV pool. Built here so its checkpoint weights load
        # (fused_wkv_wgate / ape / norm). The dual-cache attention combine lives
        # in TTHCAAttentionBackendImpl.forward_dual_decode + hca_dual_attention.
        self.compressor = None
        if self.compress_ratio == 128:
            self.compressor = TTHCACompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=hidden_size,
                head_dim=head_dim,
                prefix=f"{prefix}.compressor",
            )
        elif self.compress_ratio == 4 or mla_modules.indexer is not None:
            logger.warning_once(
                "[TT] DeepSeek v4 C4A layer (compress_ratio=4) running the HCA "
                "latent path without the sparse indexer / overlapping compressor; "
                "only the C128A high-compression dual cache is modelled on TT."
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The TT model runner passes hidden_states as 3D [users, S, H] and
        # positions as 2D [users, S]; flatten to vLLM's 2D contract and reshape
        # the output back so downstream layers see the same shape.
        orig_ndim = hidden_states.ndim
        if orig_ndim == 3:
            orig_users, orig_S, hidden_size = hidden_states.shape
            hidden_states = hidden_states.reshape(orig_users * orig_S, hidden_size)
            positions = positions.reshape(-1)
        num_tokens = hidden_states.shape[0]

        # -- MLA preprocess ------------------------------------------------
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = self.q_norm(qr)
        kv = self.kv_norm(kv)
        q = self.wq_b(qr).view(num_tokens, self.n_local_heads, self.head_dim)
        q = self.q_head_norm(q)

        # -- RoPE (GPT-J, trailing rope_head_dim dims). kv is single-head, so
        #    add a head axis for the broadcast then drop it. ------------------
        q, kv = self.rotary_emb.forward_native(positions, q, kv.unsqueeze(1))
        kv = kv.squeeze(1)

        # -- HCA attention (writes into o) ---------------------------------
        # The impl writes the vLLM output contract [tokens, n_heads * v_head_dim].
        o = torch.empty(
            num_tokens,
            self.n_local_heads * self.head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        if self.mla_attn.dual:
            # C128A: runner-paged dual cache (window + compressed pool). The
            # compressor projects from the (pre-rope) hidden states; q/kv are
            # already roped.
            self.mla_attn.forward_dual(
                q, kv, hidden_states, self.compressor, self.rotary_emb, o
            )
        else:
            self.mla_attn(q, kv, o)

        # -- O projection: inverse RoPE on the output's rope dims, then the
        #    grouped low-rank projection (wo_a) + wo_b. ----------------------
        o = o.view(num_tokens, self.n_local_heads, self.head_dim)
        o, _ = self.rotary_emb.forward_native(positions, o, None, inverse=True)
        o = o.reshape(num_tokens, self.n_local_groups, -1)
        wo_a_w = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        z = torch.einsum("bgd,grd->bgr", o, wo_a_w)
        out = self.wo_b(z.flatten(1))

        if orig_ndim == 3:
            out = out.reshape(orig_users, orig_S, out.shape[-1])
        return out
