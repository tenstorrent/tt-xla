# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# `SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""
MLA (Multi-head Latent Attention) backend for TT devices.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.v1.attention.backend import AttentionBackend, AttentionLayer, MLAAttentionImpl
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

from ..logger import tt_init_logger
from .attention import TTAttentionMetadataBuilder, TTMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = tt_init_logger(__name__)


class _TTNoopMLAPrefillBackend(MLAPrefillBackend):
    """No-op MLA prefill backend used only to satisfy MLAAttention init on TT.

    TT's OOT MLA path calls ``impl.forward(...)`` directly and does not use
    vLLM's prefill backend object. This class prevents constructor-time backend
    assertions on platforms where FlashAttention MLA prefill is unavailable.
    """

    @staticmethod
    def get_name() -> str:
        return "TT_NOOP_MLA_PREFILL"

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "TT no-op MLA prefill backend should not be executed. "
            "TTMLAAttention routes prefill via TT ops directly."
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "TT no-op MLA prefill backend should not be executed. "
            "TTMLAAttention routes prefill via TT ops directly."
        )


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class TTMLAAttentionBackend(AttentionBackend):
    """vLLM attention backend for MLA on TT devices."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @staticmethod
    def get_impl_cls() -> type["TTMLAAttentionBackendImpl"]:
        return TTMLAAttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        # Reuse the same stub builder used by the non-MLA backend; MLA does
        # not need a different metadata class for the prefill-only scope.
        return TTAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # MLA stores a single concatenated latent KV tensor per slot.
        # num_kv_heads is always 1 and head_size = kv_lora_rank + qk_rope_head_dim.
        assert num_kv_heads == 1, "num_kv_heads must be 1 for MLA"
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
        raise RuntimeError("swap_blocks is not used for the TT MLA backend.")


# --------------------------------------------------------------------------- #
# Impl
# --------------------------------------------------------------------------- #
class TTMLAAttentionBackendImpl(MLAAttentionImpl):
    """
    MLA attention impl for TT.
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

        # DeepSeek Sparse Attention (DSA). The indexer publishes this step's
        # selected key indices; `None` means dense attention for this bucket.
        self.indexer = indexer
        self.dsa_k_chunk_size = getattr(indexer, "k_chunk_size", 128)
        if indexer is not None:
            self._warn_if_head_count_blocks_kernel()

        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes are not supported for MLA on TT.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                f"Quantized MLA KV cache ({kv_cache_dtype}) is not yet "
                "supported on TT."
            )

    # ------------------------------------------------------------------ #
    # Abstract stubs — never called because we bypass forward_impl via
    # the OOT TTMLAAttention layer override below.
    # ------------------------------------------------------------------ #
    def forward_mha(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "TTMLAAttentionBackendImpl.forward_mha should never be called; "
            "the TT MLA layer routes through forward() directly. Did the "
            "OOT TTMultiHeadLatentAttentionWrapper fail to register?"
        )

    def forward_mqa(self, *args, **kwargs):
        raise RuntimeError(
            "TTMLAAttentionBackendImpl.forward_mqa should never be called; "
            "the TT MLA layer routes through forward() directly. Did the "
            "OOT TTMultiHeadLatentAttentionWrapper fail to register?"
        )

    # ------------------------------------------------------------------ #
    # Unified forward — Handles both prefill and decode here
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
        """MLA attention on TT (prefill and paged decode).
        Dispatches on token count per user: prefill (S > 1) attends against the
        freshly built local latent K via tt::flash_mla_prefill; decode (S == 1)
        attends against the paged latent KV cache via tt::paged_flash_mla_decode
        (see ``_forward_decode``).
        Shapes (from `TTMLAAttention.forward` after splitting `q`):
            q_nope:      [tokens, num_heads, qk_nope_head_dim]
            q_pe:        [tokens, num_heads, qk_rope_head_dim]
            kv_c_normed: [tokens, kv_lora_rank]
            k_pe:        [tokens, 1, qk_rope_head_dim]
            kv_cache:    [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
            output:      [tokens, num_heads * v_head_dim]   (write target)
        Returns the written output tensor.
        """
        assert (
            output is not None
        ), "TTMLAAttentionBackendImpl.forward requires an output tensor."
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
        V = self.v_head_dim
        P = self.qk_nope_head_dim

        # -- 1. Reshape inputs to [users, S, ...] --------------------------
        q_nope = q_nope.view(users, S, N, P)
        q_pe = q_pe.view(users, S, N, R)
        kv_c = kv_c_normed.view(users, S, L)
        k_pe_v = k_pe.view(users, S, 1, R)

        # -- 2. Q absorption: q_nope @ W_UK_T  ----------------------------
        # layer.W_UK_T : [num_heads, qk_nope_head_dim, kv_lora_rank].
        # W_UK_T is assigned as a plain tensor attribute (not nn.Parameter
        # or registered buffer) in MLAAttention.process_weights_after_loading
        # (mla_attention.py:797), so `model.to('xla')` doesn't move it —
        # explicit `.to(device=q_nope.device)` is required here.
        act_dtype = q_pe.dtype
        device = q_nope.device
        q_nope_lat = torch.einsum(
            "bsnp,npl->bsnl",
            q_nope,
            layer.W_UK_T.to(device=device),
        ).to(act_dtype)

        # -- 3. Build concatenated latent Q / K ---------------------------
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
    # DeepSeek Sparse Attention helpers
    # ------------------------------------------------------------------ #
    def _dsa_topk_indices(self, layer) -> Optional[torch.Tensor]:
        """This step's DSA key indices, or ``None`` for dense attention.

        The indexer runs earlier in the same forward (from
        ``MultiHeadLatentAttentionWrapper.forward``) and stashes its result on
        itself. ``layer.indexer`` is the object vLLM's ``MLAAttention.__init__``
        stored and is the same instance the wrapper holds, so no extra plumbing is
        needed; ``self.indexer`` is the fallback for direct-impl tests.
        """
        indexer = getattr(layer, "indexer", None) or self.indexer
        return getattr(indexer, "topk_indices", None) if indexer is not None else None

    def _forward_decode_sparse(
        self,
        q_lat: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        topk_indices: torch.Tensor,
        users: int,
    ) -> torch.Tensor:
        """DSA decode over only the indexer-selected keys. Returns [users, N, L].

        Restricting the *paged* decode kernel with a mask is not an option:
        ``ttnn::prim::sdpa_decode`` asserts ``is_causal``
        (``sdpa_decode_device_operation.cpp:28``, "Multi-latent attention decode
        only tested for causal!"), so a non-causal lowering aborts at runtime. And
        ``tt.sparse_sdpa`` cannot read the paged cache directly: its ``kv`` operand
        is forced ROW_MAJOR by a tt-mlir workaround while KV-cache tensors stay
        TILE, and tt-mlir does not expose tt-metal's ``cache_batch_idx`` (the
        runtime hardcodes ``std::nullopt``).

        So this gathers each user's latent cache into logical order and runs
        ``tt.sparse_sdpa`` with a single query token. That is **correctness-first,
        not fast**: the gather reads and writes the whole context, which costs more
        than dense decode's single read. Exposing ``cache_batch_idx`` on
        ``TTNN_SparseSdpaOp`` would remove the gather entirely and is the real fix.
        """
        head_dim = kv_cache.shape[-1]
        block_size = kv_cache.shape[-2]
        max_seq_len = attn_metadata.page_table.shape[1] * block_size
        page_table = attn_metadata.page_table

        outs = []
        for u in range(users):
            # Undo paging for this user: page_table[u] lists its physical blocks in
            # logical order, so a gather + reshape yields contiguous positions.
            blocks = torch.index_select(kv_cache, 0, page_table[u].to(torch.int64))
            kv_u = blocks.reshape(1, 1, max_seq_len, head_dim)  # [1, 1, T, L+R]
            outs.append(
                torch.ops.tt.sparse_sdpa(
                    # q_lat[u] is [1, N, L+R] (S == 1) -> [1, N, 1, L+R]
                    query=q_lat[u].transpose(0, 1).unsqueeze(0),
                    kv=kv_u,
                    indices=topk_indices[u : u + 1],  # [1, 1, 1, TOPK]
                    v_dim=self.kv_lora_rank,
                    scale=self.scale,
                    k_chunk_size=self.dsa_k_chunk_size,
                )  # [1, N, 1, L]
            )
        # [1, N, 1, L] per user -> [users, N, L]
        return torch.cat(outs, dim=0).squeeze(2)

    def _warn_if_head_count_blocks_kernel(self) -> None:
        """Warn when the head count vetoes tt.sparse_sdpa kernel promotion.

        ``tt.sparse_sdpa`` needs ``heads >= 32`` and ``heads % 32 == 0`` *per
        device* (post-Shardy). The op wrapper only sees the global count, so it
        cannot check this, and a violation degrades silently to the decomposition.

        Called from ``__init__``, never from ``forward``: forward runs inside the
        traced graph, where dynamo only tolerates the logging methods registered via
        ``ignore_logging_methods`` (``logger.info``) -- a ``logger.warning`` there
        aborts compilation of the whole model. The head count is static anyway.
        """
        if self.num_heads % 32 == 0 and self.num_heads >= 32:
            return
        logger.warning(
            "[TT] DSA: %d query heads is not >= 32 and a multiple of 32, so "
            "tt.sparse_sdpa cannot use its TTNN kernel and will fall back to the "
            "primitive decomposition. Note this check sees the GLOBAL head count; "
            "under tensor parallelism it is heads/model_axis_size that must satisfy "
            "the constraint.",
            self.num_heads,
        )

    @staticmethod
    def _infer_is_prefill(
        q_nope: torch.Tensor, attn_metadata: TTMetadata | None
    ) -> bool:
        """
        Prefill when more than one token per user, decode otherwise.
        Note: the scheduler guarantees that the tensors being sent to this class
        consist of either only ALL prefill requests, or ALL decode requests.
        """
        if attn_metadata is None or attn_metadata.cache_position is None:
            # Treat profiling runs as prefill
            return True
        users = attn_metadata.cache_position.shape[0]
        assert users > 0, "Invalid number of users"
        total_tokens = q_nope.shape[0]
        return (total_tokens // users) > 1

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

        topk_indices = self._dsa_topk_indices(layer)
        if topk_indices is not None:
            # DSA sparse prefill. `k_for_kernel` is already [b, 1, S, L+R], which is
            # exactly tt.sparse_sdpa's `kv` layout, and v_dim=kv_lora_rank slices the
            # latent columns W_UV expects -- so the output shape is identical to
            # flash_mla_prefill's and everything downstream is shared.
            # tt.sparse_sdpa requires batch == 1, so loop users (a compile-time
            # unroll of a bucket-static count, like the paged_fill_cache loop below).
            out_lat = torch.cat(
                [
                    torch.ops.tt.sparse_sdpa(
                        query=q_for_kernel[u : u + 1],  # [1, N, S, L+R]
                        kv=k_for_kernel[u : u + 1],  # [1, 1, S, L+R]
                        indices=topk_indices[u : u + 1],  # [1, 1, S, TOPK]
                        v_dim=self.kv_lora_rank,
                        scale=self.scale,
                        k_chunk_size=self.dsa_k_chunk_size,
                    )
                    for u in range(users)
                ],
                dim=0,
            )  # [b, N, S, L]
        else:
            out_lat = torch.ops.tt.flash_mla_prefill(
                query=q_for_kernel,
                key=k_for_kernel,
                head_dim_v=self.kv_lora_rank,
                value=None,
                attn_mask=(
                    attn_metadata.attn_mask if attn_metadata is not None else None
                ),
                is_causal=(
                    attn_metadata.is_causal if attn_metadata is not None else True
                ),
                scale=self.scale,
            )  # [b, N, S, L]

        # Expand latent output back to v_head_dim
        out = torch.einsum(
            "bnsl,nlv->bnsv",
            out_lat,
            layer.W_UV.to(device=device),
        ).to(
            act_dtype
        )  # [b, N, S, V]

        # Reshape to vLLM's output contract: [tokens, N * V]
        out = out.transpose(1, 2).reshape(
            users * seq_len, self.num_heads * self.v_head_dim
        )

        # Persist tokens in latent KV cache (this step is skipped during profiling runs)
        if (
            attn_metadata is not None
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.numel() > 0
        ):
            k_lat_for_fill = k_lat.transpose(1, 2)  # [b, 1, S, L+R]
            fill_page_table = attn_metadata.fill_page_table
            # Accumulate the per-user fills into a separate local so `kv_cache`
            # keeps referencing the bound buffer (the loop must not rebind it).
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
        """
        Paged MLA decode on TT (one token per user, S = 1).
        Shapes:
            q_lat:    [users, 1, N, L+R]                latent query (S == 1)
            k_lat:    [users, 1, 1, L+R]                new token's latent K
            kv_cache: [num_blocks, 1, block_size, L+R]  paged latent cache
        """
        # Write new token's latent K into the paged cache at the current position
        # (Skipped during profiling runs)
        if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() > 0:
            k_lat_for_update = k_lat.transpose(0, 1)  # [1, users, 1, L+R]
            updated_cache = torch.ops.tt.paged_update_cache(
                kv_cache,
                k_lat_for_update,
                attn_metadata.cache_position,
                attn_metadata.page_table,
            )
            kv_cache.copy_(updated_cache)

        # Call paged MLA decode kernel.
        # It expects query tensor to be of shape [1, users, N, L+R] and reads K/V
        # straight from the paged cache.
        #
        # DSA decode. Dense paged decode is the default and, whenever the bucket
        # cannot exceed index_topk, is not an approximation at all: top-k selects
        # every causally visible key, so it computes exactly the sparse result (see
        # dsa_decode_uses_sparse). Above that the indexer publishes indices and the
        # sparse path below runs.
        topk_indices = self._dsa_topk_indices(layer)
        is_causal = attn_metadata.is_causal if attn_metadata is not None else True
        if topk_indices is not None:
            out_lat = self._forward_decode_sparse(
                q_lat, kv_cache, attn_metadata, topk_indices, users
            )  # [users, N, L]
        else:
            out_lat = torch.ops.tt.paged_flash_mla_decode(
                query=q_lat.transpose(0, 1),  # [1, users, N, L+R]
                key=kv_cache,
                head_dim_v=self.kv_lora_rank,
                page_table=attn_metadata.page_table,
                value=None,
                is_causal=is_causal,
                attn_mask=None if is_causal else attn_metadata.attn_mask,
                cur_pos_tensor=attn_metadata.cache_position,
                scale=self.scale,
            )  # [1, users, N, L]

        # Expand latent output back to v_head_dim
        out_lat = out_lat.reshape(users, self.num_heads, self.kv_lora_rank)
        out = torch.einsum(
            "bnl,nlv->bnv",
            out_lat,
            layer.W_UV.to(device=device),
        ).to(
            act_dtype
        )  # [users, N, V]

        # Reshape to vLLM's output contract: [tokens, N * V]
        out = out.reshape(users, self.num_heads * self.v_head_dim)
        output.copy_(out)


class TTMLAAttention(MLAAttention):
    """`MLAAttention` subclass that calls `impl.forward(...)` directly."""

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        # Split q into (q_nope, q_pe). vLLM's standard MLAAttention.forward
        # only does this inside forward_impl's MQA branch; we do it here so
        # the impl sees the same tuple shape PallasMLAttentionBackendImpl
        # expects.
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(self.layer_name)
        kv_cache = self.kv_cache

        if output_shape is None:
            output_shape = (q.shape[0], self.num_heads * self.v_head_dim)
        output = torch.empty(output_shape, dtype=q.dtype, device=q.device)

        self.impl.forward(
            q=(q_nope, q_pe),
            kv_c_normed=kv_c_normed,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            layer=self,
            output=output,
        )
        return output


# OOT wrapper replacement
@MultiHeadLatentAttentionWrapper.register_oot
class TTMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    def __init__(self, *args, **kwargs):
        import vllm.model_executor.layers.attention.mla_attention as _mla_attn_module
        import vllm.model_executor.layers.mla as _mla_module

        orig_cls = _mla_module.MLAAttention
        orig_get_mla_prefill_backend = getattr(
            _mla_attn_module, "get_mla_prefill_backend", None
        )
        _mla_module.MLAAttention = TTMLAAttention
        if orig_get_mla_prefill_backend is not None:
            _mla_attn_module.get_mla_prefill_backend = (
                lambda _vllm_config: _TTNoopMLAPrefillBackend
            )
        try:
            super().__init__(*args, **kwargs)
        finally:
            _mla_module.MLAAttention = orig_cls
            if orig_get_mla_prefill_backend is not None:
                _mla_attn_module.get_mla_prefill_backend = orig_get_mla_prefill_backend
        logger.info(
            "[TT] Installed TTMLAAttention (prefix=%s) — MLA prefill uses "
            "torch.ops.tt.flash_mla_prefill; decode uses "
            "torch.ops.tt.paged_flash_mla_decode.",
            getattr(self, "prefix", "?"),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The TT model runner passes hidden_states as 3D [users, S, H] and
        # positions as 2D [users, S].
        # Flatten to vLLM's standard 2D `[total_tokens, hidden]` before the
        # upstream forward, then reshape the output back so downstream layers
        # see the same shape they sent in.
        orig_ndim = hidden_states.ndim
        if orig_ndim == 3:
            orig_users, orig_S, hidden_size = hidden_states.shape
            hidden_states = hidden_states.reshape(orig_users * orig_S, hidden_size)
            positions = positions.reshape(-1)

        out = super().forward(positions, hidden_states, llama_4_scaling)

        if orig_ndim == 3:
            out = out.reshape(orig_users, orig_S, out.shape[-1])
        return out
