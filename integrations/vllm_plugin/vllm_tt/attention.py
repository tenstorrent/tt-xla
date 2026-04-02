# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch_xla.core.xla_builder as xb
import torch_xla.experimental.custom_kernel  # noqa: F401

# Required to register custom ops.
from torch.library import impl

# from torch_xla._internal.jax_workarounds import requires_jax
from torch_xla.experimental.custom_kernel import XLA_LIB
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)

from .logger import tt_init_logger

logger = tt_init_logger(__name__)

# TT requires the head size to be a multiple of 32.
TT_HEAD_SIZE_ALIGNMENT = 32

# Note: TPU can fp8 as storage dtype but doesn't support converting from uint8
# from to fp32 directly. That's why it has a dtype mapping different from GPU
TPU_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

torch._dynamo.config.reorderable_logging_functions.add(print)


class TTAttentionMetadataBuilder:
    """
    Builder class for TT attention metadata.
    This is required by vLLM 0.13.0's encoder-only attention layer.

    The TT backend doesn't actually use the builder pattern in the same way
    as other backends, so this is a compatibility shim that returns TTMetadata
    objects when requested.
    """

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ):
        """
        Build attention metadata for TT backend.

        Returns a TTMetadata instance. Note that the actual metadata construction
        happens elsewhere in the TT backend's pipeline, so this returns a minimal
        stub that will be replaced during actual execution.
        """
        # Return a minimal TTMetadata stub
        # The actual metadata will be constructed in the model runner
        return TTMetadata(
            cache_position=None,
            attn_mask=None,
            page_table=None,
            is_causal=getattr(common_attn_metadata, "causal", True),
        )


class TTAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["TTAttentionBackendImpl"]:
        return TTAttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        # Return the stub builder class for encoder-only attention support
        return TTAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    # In recent TPU generations, up to v6e, the SMEM size is 1MB. The
    # block_tables within the TTMetadata constitute almost the entire SMEM
    # requirement. Its size is max_num_seqs * num_page_per_seq * 4 (Int). Here
    # we simply make sure that the size is smaller than half of SMEM capacity.
    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        max_num_page_per_req = (
            1024 * 1024 // 2 // vllm_config.scheduler_config.max_num_seqs // 4
        )
        min_page_size = cdiv(
            vllm_config.model_config.max_model_len, max_num_page_per_req
        )
        min_page_size = 1 << (min_page_size - 1).bit_length()
        return min_page_size

    @staticmethod
    def get_max_num_seqs(model_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(model_len, page_size)
        return 1024 * 1024 // 2 // num_page_per_req // 4

    # TPU has limited SREGs (scalar registers), if page_size is too small, we
    # can spill SREGs easily which leads to bad performance. The strategy we
    # apply here is trying to split max-model-len to 16 pages which make the
    # spill less likely. Meanwhile we make sure the page size is in [16, 256].
    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        # TODO: This is a temporary fix for vmem OOM.
        # For long model length, we use 16 page-size to avoid too much
        # VMEM spill. A more robust solution should be implemented to
        # handle VREG spills.
        return 32
        if vllm_config.model_config.max_model_len > 8192:
            return 16
        page_size = next_power_of_2(vllm_config.model_config.max_model_len) // 16
        if page_size <= 16:
            return 16
        if page_size >= 256:
            return 256
        return page_size


# ttnn.fill_cache has a limitation. If the work that needs to be done to fill the cache does not fit on the device grid,
# it will fail to compile the op. This workaround pads the fill value to the same shape as the cache and we use this new
# tensor as the cache instead.
#
# This is functionally the same as a fill_cache op, but this avoids the limitation of ttnn.fill_cache.
def fill_cache_workaround(
    cache_shape: List[int], fill_value: torch.Tensor
) -> torch.Tensor:
    new_cache = torch.nn.functional.pad(
        fill_value, (0, 0, 0, cache_shape[-2] - fill_value.shape[-2], 0, 0, 0, 0)
    )
    return new_cache


@dataclass
class TTMetadata:
    # Used in the TTAttentionBackendImpl
    cache_position: torch.Tensor
    attn_mask: torch.Tensor
    page_table: torch.Tensor
    is_causal: bool

    def __init__(
        self,
        cache_position: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        is_causal: bool = True,
    ):
        self.cache_position = cache_position
        self.attn_mask = attn_mask
        self.page_table = page_table
        self.is_causal = is_causal


class TTAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[int] = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")

        if attn_type not in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        ):
            raise NotImplementedError(
                f"TT attention only supports encoder or decoder attention, but got {attn_type}."
            )

        self.kv_cache_stored = None

        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(
                kv_cache_dtype.lower().strip()
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

    # @torch.compiler.disable
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with TT attention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [num_blocks, block_size, num_kv_heads * 2, head_size]
                    - now [2, batch_size, max_seq_len, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]

        query/key/value/output tensors have additional dimension 'batch_size'
        for batched inputs.
            query: shape = [batch_size, num_tokens, num_heads * head_size]
            key: shape = [batch_size, num_tokens, num_kv_heads * head_size]
            value: shape = [batch_size, num_tokens, num_kv_heads * head_size]
            output: shape = [batch_size, num_tokens, num_heads * head_size]
        """

        # Prepare inputs and metadata
        inputs = self._prepare_inputs(query, key, value, attn_metadata)

        # Handle paged attention if KV cache exists
        if kv_cache.numel() > 1:
            self._handle_paged_attention(inputs, kv_cache, attn_metadata)

        # Compute attention based on mode:
        # - is_prefill=True: Full attention (prefill phase for generative models,
        #                    or single-pass attention for pooling models)
        # - is_prefill=False: Paged decode attention (generative models only)
        if inputs.is_prefill:
            assert self.sinks is None, "Attention sink is unsupported in SDPA prefill"
            output = self._compute_full_attention(inputs, attn_metadata)
        else:
            output = self._compute_decode_attention(inputs, kv_cache, attn_metadata)

        # Finalize output shape to match original input
        return self._finalize_output(output, inputs)

    def _prepare_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TTMetadata,
    ):
        """Prepare and reshape input tensors for attention computation."""
        from collections import namedtuple

        # Extract common metadata
        # Handle case when cache_position is None (e.g., during profiling)
        if attn_metadata is not None and attn_metadata.cache_position is not None:
            num_users = attn_metadata.cache_position.shape[0]  # logical batch size
        else:
            # Fallback: infer from query shape
            num_users = query.shape[0] if query.ndim > 2 else 1

        orig_query_shape = query.shape
        orig_query_ndim = query.ndim

        if orig_query_ndim == 3:
            assert query.shape[0] == num_users, (
                f"query batch dim ({query.shape[0]}) and cache_position num_users "
                f"({num_users}) mismatch."
            )
            assert (
                key.shape == value.shape
            ), "key and value shape mismatch for batched inputs."
        elif orig_query_ndim == 2:
            # Reshape query to [users, tokens_per_user, hidden_size]
            query = self._reshape_query(query, num_users)
            key, value = self._reshape_key_value(key, value, num_users)
        else:
            raise ValueError(
                f"Unsupported query ndim: {orig_query_ndim}, expected 2 or 3."
            )

        users_kv = key.shape[0]
        query_num_tokens = query.shape[1]
        kv_num_tokens = key.shape[1]
        hidden_size = query.shape[2]

        # Determine prefill vs decode mode
        is_prefill = query_num_tokens > 1

        # Reshape Q/K/V to [batch(users), tokens, num_heads, head_size]
        query, key, value = self._reshape_to_attention_format(
            query,
            key,
            value,
            num_users,
            users_kv,
            query_num_tokens,
            kv_num_tokens,
            hidden_size,
        )

        # Create named tuple for inputs
        AttentionInputs = namedtuple(
            "AttentionInputs",
            [
                "query",
                "key",
                "value",
                "orig_query_shape",
                "orig_query_ndim",
                "users",
                "query_num_tokens",
                "is_prefill",
                "users_kv",
                "kv_num_tokens",
            ],
        )

        return AttentionInputs(
            query=query,
            key=key,
            value=value,
            orig_query_shape=orig_query_shape,
            orig_query_ndim=orig_query_ndim,
            users=num_users,
            query_num_tokens=query_num_tokens,
            is_prefill=is_prefill,
            users_kv=users_kv,
            kv_num_tokens=kv_num_tokens,
        )

    def _reshape_query(self, query: torch.Tensor, num_users: int):
        """Reshape query tensor to [users, tokens_per_user, hidden] format."""
        # [total_tokens, hidden] (vLLM style)
        total_tokens = query.shape[0]
        hidden_size = query.shape[1]
        users = num_users
        assert (
            total_tokens % users == 0
        ), f"total_tokens ({total_tokens}) not divisible by num_users ({users})."
        query_num_tokens = total_tokens // users  # tokens per user
        query = query.view(users, query_num_tokens, hidden_size)

        return query

    def _reshape_key_value(self, key: torch.Tensor, value: torch.Tensor, users: int):
        """Reshape key and value tensors to [users_kv, kv_num_tokens, hidden] format."""
        total_k_tokens = key.shape[0]
        kv_hidden_size = key.shape[1]
        users_kv = users  # Assume same users as query
        assert (
            total_k_tokens % users_kv == 0
        ), f"total_k_tokens ({total_k_tokens}) not divisible by users_kv ({users_kv})."
        kv_num_tokens = total_k_tokens // users_kv
        key = key.view(users_kv, kv_num_tokens, kv_hidden_size)

        total_v_tokens = value.shape[0]
        v_hidden_size = value.shape[1]
        users_v = users_kv
        assert (
            total_v_tokens % users_v == 0
        ), f"total_v_tokens ({total_v_tokens}) not divisible by users_v ({users_v})."
        v_num_tokens = total_v_tokens // users_v
        assert v_num_tokens == kv_num_tokens, "key/value token count mismatch."
        value = value.view(users_v, v_num_tokens, v_hidden_size)

        return key, value

    def _reshape_to_attention_format(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        users: int,
        users_kv: int,
        query_num_tokens: int,
        kv_num_tokens: int,
        hidden_size: int,
    ):
        """Reshape Q/K/V tensors to [batch(users), tokens, num_heads, head_size] format."""
        num_heads = hidden_size // self.head_size
        assert hidden_size % self.head_size == 0

        query = query.reshape(users, query_num_tokens, num_heads, self.head_size)
        key = key.reshape(
            users_kv, kv_num_tokens, key.shape[-1] // self.head_size, self.head_size
        )
        value = value.reshape(
            users_kv, kv_num_tokens, value.shape[-1] // self.head_size, self.head_size
        )

        return query, key, value

    def _handle_paged_attention(
        self, inputs, kv_cache: torch.Tensor, attn_metadata: TTMetadata
    ):
        """Handle paged attention cache updates."""
        k_cache = kv_cache[0]
        v_cache = kv_cache[1]

        if not inputs.is_prefill:
            # Decode: update single token in cache
            key_for_update = inputs.key.transpose(0, 1)
            value_for_update = inputs.value.transpose(0, 1)

            k_cache = torch.ops.tt.paged_update_cache(
                k_cache,
                key_for_update,
                attn_metadata.cache_position,
                attn_metadata.page_table,
            )
            v_cache = torch.ops.tt.paged_update_cache(
                v_cache,
                value_for_update,
                attn_metadata.cache_position,
                attn_metadata.page_table,
            )
        else:
            # Prefill: fill multiple tokens at once
            key_for_update = inputs.key.transpose(1, 2)
            value_for_update = inputs.value.transpose(1, 2)

            for batch_idx in range(inputs.users):
                k_cache = torch.ops.tt.paged_fill_cache(
                    k_cache,
                    key_for_update[batch_idx : batch_idx + 1],
                    attn_metadata.page_table,
                    batch_idx=torch.tensor(
                        [batch_idx], dtype=torch.int32, device=k_cache.device
                    ),
                )
                v_cache = torch.ops.tt.paged_fill_cache(
                    v_cache,
                    value_for_update[batch_idx : batch_idx + 1],
                    attn_metadata.page_table,
                    batch_idx=torch.tensor(
                        [batch_idx], dtype=torch.int32, device=v_cache.device
                    ),
                )

        # Update the KV cache
        new_kv_cache = torch.stack([k_cache, v_cache], dim=0)
        kv_cache.copy_(new_kv_cache)

    def _compute_full_attention(
        self, inputs, attn_metadata: TTMetadata
    ) -> torch.Tensor:
        """Compute full attention using scaled dot-product attention (non-paged).

        This method is used in two scenarios:
        1. Generative models: During the prefill phase when processing initial
           prompt tokens before decode iterations begin.
        2. Pooling models: For the entire attention computation, as these models
           process all tokens in a single pass without a decode phase or KV cache.
        """
        # scaled_dot_product_attention expects [B, N_tokens, N_heads, H]
        query_for_sdpa = inputs.query.transpose(-3, -2)
        key_for_sdpa = inputs.key.transpose(-3, -2)
        value_for_sdpa = inputs.value.transpose(-3, -2)

        output = torch.ops.tt.scaled_dot_product_attention(
            query_for_sdpa,
            key_for_sdpa,
            value_for_sdpa,
            is_causal=attn_metadata.is_causal,
            attn_mask=attn_metadata.attn_mask,
        ).transpose(
            -3, -2
        )  # Back to [users, tokens, num_heads, head_size]

        return output

    def _compute_decode_attention(
        self, inputs, kv_cache: torch.Tensor, attn_metadata: TTMetadata
    ) -> torch.Tensor:
        """Compute attention for decode phase (paged)."""
        k_cache = kv_cache[0]
        v_cache = kv_cache[1]

        # Adjust for decode kernel expecting query as [1, num_users, num_heads, head]
        # Current query: [users, query_num_tokens, num_heads, head_size]
        # In decode, query_num_tokens == 1 is normal
        query_for_decode = inputs.query.transpose(0, 1)

        out = torch.ops.tt.paged_scaled_dot_product_attention_decode(
            query_for_decode,
            k_cache,
            v_cache,
            attn_metadata.page_table,
            cur_pos_tensor=attn_metadata.cache_position,
            is_causal=attn_metadata.is_causal,
            attn_mask=attn_metadata.attn_mask,
            attention_sink=self.sinks,
        )
        # out: [query_num_tokens, users, num_heads, head_size]
        out = out.transpose(0, 1)  # [users, query_num_tokens, num_heads, head_size]

        return out

    def _finalize_output(self, output: torch.Tensor, inputs) -> torch.Tensor:
        """Finalize output shape to match original input dimensions."""
        hidden_size = inputs.orig_query_shape[-1]

        # Output from both prefill and decode: [users, tokens, num_heads, head_size]
        if inputs.orig_query_ndim == 3:
            return output.reshape(inputs.users, inputs.query_num_tokens, hidden_size)
        else:
            total_tokens = inputs.users * inputs.query_num_tokens
            return output.reshape(total_tokens, hidden_size)


class TTMLAAttentionBackend(AttentionBackend):
    """
    Attention backend for MLA (Multi-Latent Attention) models such as DeepSeek V3/R1.

    In MLA the KV cache stores the concatenated compressed latent and RoPE key:
        [num_blocks, num_kv_heads=1, block_size, kv_lora_rank + qk_rope_head_dim]

    The V projection (W_UV) is applied post-attention by the outer MLAAttention layer
    via _v_up_proj, so the kernel only needs to return kv_lora_rank-dimensional outputs.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM_MLA"

    @staticmethod
    def get_impl_cls() -> type["TTMLAAttentionBackendImpl"]:
        return TTMLAAttentionBackendImpl

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
        # MLA uses a single K-only cache; num_kv_heads is always 1 and
        # head_size = kv_lora_rank + qk_rope_head_dim.
        return (num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TT MLA backend.")

    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        return TTAttentionBackend.get_min_page_size(vllm_config)

    @staticmethod
    def get_max_num_seqs(model_len: int, page_size: int) -> int:
        return TTAttentionBackend.get_max_num_seqs(model_len, page_size)

    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        return TTAttentionBackend.get_page_size(vllm_config)


class TTMLAAttentionBackendImpl(MLAAttentionImpl):
    """
    Inner attention implementation for MLA (Multi-Latent Attention) models.

    Plugged in as ``self.impl`` inside vLLM's ``MLAAttention`` layer.  The outer
    layer handles KV-cache writes (concat_and_cache_mla), W_UK query absorption,
    and the post-attention W_UV up-projection (_v_up_proj).  This class only
    needs to implement the two attention kernels:

    * ``forward_mha`` – prefill, calls ``torch.ops.tt.flash_mla_prefill``
    * ``forward_mqa`` – decode,  calls ``torch.ops.tt.paged_flash_multi_latent_attention_decode``

    Tensor conventions (matching the outer MLAAttention.forward_impl):
      forward_mha inputs:
        q             : [T, num_heads, qk_head_dim]
        kv_c_normed   : [T, kv_lora_rank]
        k_pe          : [T, 1, qk_rope_head_dim]
        output        : [T, num_heads * v_head_dim]  (written in-place)

      forward_mqa inputs:
        q             : (q_nope_absorbed [B,N,kv_lora_rank], q_pe [B,N,qk_rope_head_dim])
                        or combined [B, N, kv_lora_rank+qk_rope_head_dim]
        kv_cache      : [num_blocks, 1, block_size, kv_lora_rank+qk_rope_head_dim]
        returns       : ([B, N, kv_lora_rank], lse_or_None)
                        (_v_up_proj in the outer layer projects to v_head_dim)
    """

    # Required attributes checked by the outer MLAAttention layer.
    supports_quant_query_input: bool = False
    dcp_world_size: int = 1

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
        # MLA-specific arguments (forwarded verbatim from MLAAttention.__init__)
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj,
        indexer=None,
        q_pad_num_heads: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size          # = kv_lora_rank + qk_rope_head_dim
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.indexer = indexer
        self.q_pad_num_heads = q_pad_num_heads

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Prefill attention via flash_mla_prefill.

        Builds the combined latent key K = cat(kv_c_normed, k_pe) and pads
        variable-length sequences to a fixed batch before calling the kernel.
        The kernel outputs [batch, num_heads, padded_sl, kv_lora_rank]; the V
        up-projection is then applied to obtain the final v_head_dim output.
        """
        T = q.shape[0]                   # total prefill tokens
        device, dtype = q.device, q.dtype

        # Build K: [T, 1, kv_lora_rank + qk_rope_head_dim]
        k_c = kv_c_normed.unsqueeze(1)   # [T, 1, kv_lora_rank]
        k_p = k_pe if k_pe.dim() == 3 else k_pe.unsqueeze(1)  # [T, 1, qk_rope_head_dim]
        k = torch.cat([k_c, k_p], dim=-1)  # [T, 1, head_size]

        # Convert variable-length token stream → batched tensors using query_start_loc
        qsl = attn_metadata.prefill.query_start_loc  # [batch+1]
        batch_size = qsl.shape[0] - 1

        seq_lens = (qsl[1:] - qsl[:-1]).tolist()
        max_sl = max(seq_lens)
        padded_sl = ((max_sl + 31) // 32) * 32   # flash_mla_prefill needs multiple of 32

        q_b = q.new_zeros(batch_size, self.num_heads, padded_sl, self.qk_head_dim)
        k_b = q.new_zeros(batch_size, 1, padded_sl, self.head_size)

        for i in range(batch_size):
            s, e = int(qsl[i]), int(qsl[i + 1])
            sl = e - s
            q_b[i, :, :sl, :] = q[s:e].transpose(0, 1)    # [N, sl, qk_head_dim]
            k_b[i, :, :sl, :] = k[s:e].transpose(0, 1)    # [1, sl, head_size]

        # flash_mla_prefill: output [batch, num_heads, padded_sl, kv_lora_rank]
        attn_latent = torch.ops.tt.flash_mla_prefill(
            q_b, k_b, self.kv_lora_rank, is_causal=True
        )

        # Apply kv_b_proj to project latent → (k_nope, v) and extract V
        # kv_b_proj: [*, kv_lora_rank] → [*, num_heads * (qk_nope_head_dim + v_head_dim)]
        out_view = output.view(T, self.num_heads, self.v_head_dim)
        for i in range(batch_size):
            s, e = int(qsl[i]), int(qsl[i + 1])
            sl = e - s
            # attn_latent[i]: [num_heads, padded_sl, kv_lora_rank]
            # Reshape to [sl, kv_lora_rank] for projection (average heads is wrong;
            # project each head's weighted latent separately via kv_b_proj)
            latent_i = attn_latent[i, :, :sl, :].transpose(0, 1)  # [sl, N, kv_lora_rank]
            # Project each token's per-head latent through the shared kv_b_proj.
            # kv_b_proj expects [*, kv_lora_rank] input.
            latent_flat = latent_i.reshape(sl, self.num_heads, self.kv_lora_rank)
            proj = self.kv_b_proj(
                latent_flat.reshape(sl * self.num_heads, self.kv_lora_rank)
            )[0].view(sl, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            _, v = proj.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            out_view[s:e] = v

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode attention via paged_flash_multi_latent_attention_decode.

        The outer MLAAttention.forward_impl passes q as a tuple
        (q_nope_absorbed, q_pe) after the W_UK absorption step.  We
        concatenate them to build the full absorbed query.

        Returns (attn_out, None) where attn_out has shape [B, N, kv_lora_rank].
        The caller (_v_up_proj in MLAAttention.forward_impl) applies W_UV.
        """
        if isinstance(q, tuple):
            q_nope, q_pe = q   # [B, N, kv_lora_rank], [B, N, qk_rope_head_dim]
            q = torch.cat([q_nope, q_pe], dim=-1)  # [B, N, head_size]

        B, N, dh = q.shape
        q_for_op = q.unsqueeze(0)  # [1, B, N, dh]

        decode_meta = attn_metadata.decode
        page_table = decode_meta.block_table   # [B, max_blocks], int32
        seq_lens = decode_meta.seq_lens        # [B]
        cur_pos = (seq_lens - 1).to(torch.int32)

        # paged_flash_multi_latent_attention_decode:
        # output [1, B, N, kv_lora_rank]
        out = torch.ops.tt.paged_flash_multi_latent_attention_decode(
            q_for_op,
            kv_c_and_k_pe_cache,
            self.kv_lora_rank,
            page_table,
            is_causal=True,
            cur_pos_tensor=cur_pos,
            sliding_window_size=self.sliding_window,
        )
        return out.squeeze(0), None  # [B, N, kv_lora_rank]


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_slices_per_kv_cache_update_block: int,
    num_kv_update_slices: torch.Tensor,
    kv_cache_quantized_dtype: Optional[torch.dtype] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Write the key and values to the KV cache.
    Args:
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        kv_cache: shape = [num_blocks, block_size, num_kv_heads * 2, head_size]
        num_slices_per_kv_cache_update_block: int
    """
    _, page_size, num_combined_kv_heads, head_size = kv_cache.shape
    head_size = cdiv(head_size, TT_HEAD_SIZE_ALIGNMENT) * TT_HEAD_SIZE_ALIGNMENT

    if kv_cache_quantized_dtype is not None:
        dtype_info = torch.finfo(kv_cache_quantized_dtype)
        key = key.to(torch.float32) / k_scale
        # NOTE: clamp is added here to avoid out of range of quantized dtype
        key = torch.clamp(key, dtype_info.min, dtype_info.max)
        key = key.to(kv_cache_quantized_dtype)
        value = value.to(torch.float32) / v_scale
        value = torch.clamp(value, dtype_info.min, dtype_info.max)
        value = value.to(kv_cache_quantized_dtype)

    kv = torch.cat([key, value], axis=-1).reshape(-1, num_combined_kv_heads, head_size)

    torch.ops.xla.dynamo_set_buffer_donor_(kv_cache, True)

    kv_cache = kv_cache.flatten(0, 1)
    new_kv_cache = torch.ops.xla.kv_cache_update_op(
        kv,
        slot_mapping,
        kv_cache,
        num_kv_update_slices,
        page_size,
        num_slices_per_kv_cache_update_block,
    )
    # NOTE: the in-place copy will be optimized away by XLA compiler.
    kv_cache.copy_(new_kv_cache)


# We can move this function to a common utils file if it's also useful for other
# hardware.
def dtype_bits(dtype: torch.dtype):
    if dtype.is_floating_point:
        try:
            return torch.finfo(dtype).bits
        except TypeError:
            pass
    elif dtype.is_complex:
        if dtype is torch.complex32:
            return 32
        elif dtype is torch.complex64:
            return 64
        elif dtype is torch.complex128:
            return 128
    else:
        try:
            return torch.iinfo(dtype).bits
        # torch.iinfo cannot support int4, int2, bits8...
        except TypeError:
            pass
    str_dtype = str(dtype)
    # support torch.int4, torch.int5, torch.uint5...
    if str_dtype.startswith("torch.int") or str_dtype.startswith("torch.uint"):
        return int(str_dtype[-1])
    raise TypeError(f"Getting the bit width of {dtype} is not supported")


def get_dtype_packing(dtype):
    bits = dtype_bits(dtype)
    if 32 % bits != 0:
        raise ValueError(
            f"The bit width must be divisible by 32, but got bits={bits}, "
            "dtype={dtype}"
        )
    return 32 // bits


def get_page_size_bytes(
    block_size: int, num_kv_heads: int, head_size: int, kv_cache_dtype: torch.dtype
) -> int:
    """Returns the size in bytes of one page of the KV cache."""
    padded_head_size = cdiv(head_size, TT_HEAD_SIZE_ALIGNMENT) * TT_HEAD_SIZE_ALIGNMENT
    num_combined_kv_heads = num_kv_heads * 2

    # NOTE: for the implicit padding in XLA
    packing = get_dtype_packing(kv_cache_dtype)
    num_combined_kv_heads = cdiv(num_combined_kv_heads, packing) * packing

    kv_cache_dtype_bits = dtype_bits(kv_cache_dtype)
    return (
        block_size * num_combined_kv_heads * padded_head_size * kv_cache_dtype_bits // 8
    )
