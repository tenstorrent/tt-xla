# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# `SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""
MLA (Multi-head Latent Attention) backend for TT devices.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch_xla.distributed.spmd as xs
from tt_torch.sharding import sharding_constraint_tensor
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.v1.attention.backend import AttentionBackend, AttentionLayer, MLAAttentionImpl

from .attention import TTAttentionMetadataBuilder, TTMetadata
from .logger import tt_init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = tt_init_logger(__name__)


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
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
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
        ``attn_mask`` is an optional additive mask. When ``None`` (plain MLA) the
        kernels run in their built-in causal mode. When provided (DeepSeek Sparse
        Attention — see ``attention_dsa.py``) it carries the combined causal +
        top-k sparsity mask, the kernels run with ``is_causal=False`` and the mask
        is the sole source of masking. The mask is broadcast over the query heads
        (shape ``[users, 1, S, S]`` for prefill, ``[users, 1, 1, max_seq]`` for
        decode), matching the indexer's head-independent token selection.
        Returns the written output tensor.
        """
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
            return self._forward_prefill(
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
                attn_mask,
            )
        else:
            return self._forward_decode(
                q_lat,
                k_lat,
                kv_cache,
                attn_metadata,
                layer,
                users,
                act_dtype,
                device,
                output,
                attn_mask,
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
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_for_kernel = q_lat.transpose(1, 2).contiguous()  # [b, N, S, L+R]
        k_for_kernel = k_lat.transpose(1, 2).contiguous()  # [b, 1, S, L+R]

        # A sparse (DSA) mask supersedes the kernel's built-in causal masking:
        # the mask already encodes causality plus the indexer's top-k selection,
        # and tt::flash_mla_prefill forbids is_causal=True alongside a mask.
        if attn_mask is not None:
            is_causal = False
        else:
            is_causal = attn_metadata.is_causal if attn_metadata is not None else True

        out_lat = torch.ops.tt.flash_mla_prefill(
            query=q_for_kernel,
            key=k_for_kernel,
            head_dim_v=self.kv_lora_rank,
            value=None,
            attn_mask=attn_mask,
            is_causal=is_causal,
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
        layer: "MLAAttention",
        users: int,
        act_dtype: torch.dtype,
        device: torch.device,
        output: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Paged MLA decode on TT (one token per user, S = 1).
        Shapes:
            q_lat:    [users, 1, N, L+R]                latent query (S == 1)
            k_lat:    [users, 1, 1, L+R]                new token's latent K
            kv_cache: [num_blocks, 1, block_size, L+R]  paged latent cache
        ``attn_mask`` (DSA): additive mask broadcastable to
        ``[users, nqh, 1, max_seq_len]`` that already encodes causality + top-k
        sparsity; when given the kernel runs with ``is_causal=False``.
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
        if attn_mask is not None:
            # DSA: the indexer mask already encodes causality + top-k sparsity, so
            # run the kernel uncausal with the explicit mask. cur_pos is still
            # forwarded (the decode kernel ignores it for masking when not causal).
            is_causal = False
            # The paged MLA decode kernel lays the mask out as
            # [users, 1, num_heads, max_seq]: with S == 1 the query heads occupy the
            # kernel's row dimension, so the mask validation requires a head per row
            # (mask_shape[2] == q heads), unlike prefill's head-broadcast
            # [users, 1, S, S]. The indexer's top-k selection is head-independent, so
            # broadcast the single computed row across all query heads.
            decode_mask = attn_mask.expand(-1, -1, self.num_heads, -1)
            # Under tensor parallelism the query heads are sharded over the "model"
            # mesh axis (q_b_proj is column-parallel), so per device q has
            # num_heads / model_axis heads. Constrain the mask's head dim the same
            # way or the kernel sees a 128-head mask against 32-head q. The mask rows
            # are identical across heads, so any head-shard slice is correct. Use the
            # Shardy sharding-constraint op (the supported way to reshard an
            # intermediate inside the compiled graph), not xs.mark_sharding which is
            # for graph params/inputs.
            mesh = xs.get_global_mesh()
            if mesh is not None:
                decode_mask = sharding_constraint_tensor(
                    decode_mask, mesh, (None, None, "model", None)
                )
        else:
            is_causal = attn_metadata.is_causal if attn_metadata is not None else True
            decode_mask = None if is_causal else attn_metadata.attn_mask
        out_lat = torch.ops.tt.paged_flash_mla_decode(
            query=q_lat.transpose(0, 1),  # [1, users, N, L+R]
            key=kv_cache,
            head_dim_v=self.kv_lora_rank,
            page_table=attn_metadata.page_table,
            value=None,
            is_causal=is_causal,
            attn_mask=decode_mask,
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
        if output is not None:
            output.copy_(out)
            return output
        return out


class TTMLAAttention(MLAAttention):
    """`MLAAttention` subclass that calls `impl.forward(...)` directly.

    ``attn_mask`` is forwarded to the impl unchanged; it is ``None`` for plain
    MLA and carries the DSA top-k sparsity mask when the OOT wrapper runs the
    indexer (see ``attention_dsa.dsa_wrapper_forward``).
    """

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: Optional[torch.Size] = None,
        attn_mask: Optional[torch.Tensor] = None,
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
            attn_mask=attn_mask,
        )
        return output


# OOT wrapper replacement
@MultiHeadLatentAttentionWrapper.register_oot
class TTMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    def __init__(self, *args, **kwargs):
        import vllm.model_executor.layers.mla as _mla_module

        orig_cls = _mla_module.MLAAttention
        _mla_module.MLAAttention = TTMLAAttention
        try:
            super().__init__(*args, **kwargs)
        finally:
            _mla_module.MLAAttention = orig_cls
        # `is_sparse`/`indexer` are set by the upstream __init__ from
        # `mla_modules`. DeepSeek-V3.2 (DSA) sets is_sparse=True and supplies an
        # indexer; everything else is plain MLA.
        self._tt_is_sparse = bool(
            getattr(self, "is_sparse", False) and getattr(self, "indexer", None)
        )
        logger.info(
            "[TT] Installed TTMLAAttention (prefix=%s, sparse=%s) — MLA prefill "
            "uses torch.ops.tt.flash_mla_prefill; decode uses "
            "torch.ops.tt.paged_flash_mla_decode.%s",
            getattr(self, "prefix", "?"),
            self._tt_is_sparse,
            " DSA top-k masking via attention_dsa." if self._tt_is_sparse else "",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # DeepSeek Sparse Attention: bypass the upstream forward (which would
        # call the GPU-only indexer op) and run the TT indexer + sparse MLA path.
        # dsa_wrapper_forward handles the 3D<->2D reshape itself.
        if self._tt_is_sparse:
            from .attention_dsa import dsa_wrapper_forward

            return dsa_wrapper_forward(self, positions, hidden_states, llama_4_scaling)

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
