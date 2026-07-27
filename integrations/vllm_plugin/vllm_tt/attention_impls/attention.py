# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch_xla.core.xla_builder as xb
import torch_xla.experimental.custom_kernel  # noqa: F401

# Required to register custom ops.
from torch.library import impl

# from torch_xla._internal.jax_workarounds import requires_jax
from torch_xla.experimental.custom_kernel import XLA_LIB
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.encoder_only_attention import (
    create_encoder_only_attention_backend,
)
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec

from ..logger import tt_init_logger

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
        # Shape for one of the two separate K/V tensors per layer.
        return (num_blocks, num_kv_heads, block_size, head_size)

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
    # Page table with prefix blocks rolled to the end for paged_fill_cache.
    # Computed outside the compiled graph to avoid shape-change recompilation.
    fill_page_table: torch.Tensor
    # Chunked-prefill prefix offset: device [1] int32 = num_computed (shared
    # across users by same-stage batching). Set only on a cached-prefix chunk;
    # consumed by the chunked_scaled_dot_product_attention op.
    chunk_start_idx: torch.Tensor
    batch_idx: Optional[torch.Tensor] = None
    num_users: Optional[int] = None

    def __init__(
        self,
        cache_position: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        is_causal: bool = True,
        fill_page_table: torch.Tensor | None = None,
        mesh: object | None = None,
        dp_size: int = 1,
        chunk_start_idx: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        num_users: Optional[int] = None,
    ):
        self.cache_position = cache_position
        self.attn_mask = attn_mask
        self.page_table = page_table
        self.is_causal = is_causal
        self.fill_page_table = (
            fill_page_table if fill_page_table is not None else page_table
        )
        self.mesh = mesh
        # Number of batch shards; used by paged_fill_cache to rebase batch_idx
        # into per-shard local ids.
        self.dp_size = dp_size
        self.chunk_start_idx = chunk_start_idx
        self.num_users = num_users
        self.batch_idx = batch_idx


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
    ) -> None:
        """Forward pass with TT attention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [num_blocks, block_size, num_kv_heads * 2, head_size]
                    - now [2, batch_size, max_seq_len, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            Use pre-allocated output buffer to return the result.
            shape = [num_tokens, num_heads * head_size]
        """
        assert attn_metadata is not None, "TT attention requires metadata."
        assert output is not None, "TT attention requires an output buffer."
        output_buffer = output

        # Prepare inputs and metadata
        inputs = self._prepare_inputs(query, key, value, attn_metadata)

        # kv_cache is [k_cache, v_cache] after init, but a scalar placeholder
        # during profiling — isinstance distinguishes the two cases.
        # Skip cache update when sharing KV cache with another layer — the
        # target layer already wrote to the shared cache.
        if (
            isinstance(kv_cache, (list, tuple))
            and kv_cache[0].numel() > 0
            and self.kv_sharing_target_layer_name is None
        ):
            self._handle_paged_attention(inputs, kv_cache, attn_metadata)

        # Compute attention based on mode:
        # - is_prefill=True: Full attention (prefill phase for generative models,
        #                    or single-pass attention for pooling models)
        # - is_prefill=False: Paged decode attention (generative models only)
        if inputs.is_prefill:
            assert self.sinks is None, "Attention sink is unsupported in SDPA prefill"
            # Pass kv_cache so shared-KV layers can gather K/V from the target
            # layer's paged cache (see ``_compute_full_attention`` docstring).
            attn_output = self._compute_full_attention(inputs, kv_cache, attn_metadata)
        else:
            attn_output = self._compute_decode_attention(
                inputs, kv_cache, attn_metadata
            )

        # Reshape final output to match vLLM expected flattened shape
        # [num_users*num_tokens, num_heads * head_size].
        finalized_output = attn_output.reshape(-1, self.num_heads * self.head_size)

        # vLLM passes a preallocated output buffer and expects attention impls
        # to materialize results into it.
        output_buffer.copy_(finalized_output.reshape_as(output_buffer))

    def _normalize_to_attention_format(
        self,
        tensor: torch.Tensor,
        tensor_name: str,
        expected_num_heads: int,
        expected_head_size: int,
        num_users: int,
    ) -> torch.Tensor:
        """Normalize a tensor to [users, tokens, heads, head_size] format.

        The leading dimension is always treated as flattened tokens
        (batch_size * num_tokens), never as an explicit batch axis.
        """
        if tensor.ndim < 2:
            raise ValueError(
                f"{tensor_name} must have rank >= 2, got shape={tuple(tensor.shape)}"
            )

        expected_hidden = expected_num_heads * expected_head_size
        if tensor.ndim == 2 and tensor.shape[-1] == expected_hidden:
            normalized = tensor.reshape(-1, expected_num_heads, expected_head_size)
        elif (
            tensor.ndim == 3
            and tensor.shape[-2] == expected_num_heads
            and tensor.shape[-1] == expected_head_size
        ):
            normalized = tensor
        else:
            raise ValueError(
                f"{tensor_name} must be either a flattened-token tensor with "
                f"shape[-1] == {expected_hidden} or a head-shaped tensor with "
                f"shape[-2:] == ({expected_num_heads}, {expected_head_size}); "
                f"got shape={tuple(tensor.shape)}"
            )

        total_tokens = normalized.shape[0]
        if total_tokens % num_users != 0:
            raise ValueError(
                f"{tensor_name} total tokens ({total_tokens}) must be divisible "
                f"by num_users ({num_users})"
            )
        return normalized.reshape(num_users, -1, expected_num_heads, expected_head_size)

    def _prepare_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TTMetadata,
    ):
        """Prepare and reshape input tensors for attention computation."""
        from collections import namedtuple

        num_users = attn_metadata.num_users
        assert num_users is not None, "num_users must be provided in attn_metadata."
        orig_query_shape = query.shape
        orig_query_ndim = query.ndim

        query = self._normalize_to_attention_format(
            query,
            "query",
            self.num_heads,
            self.head_size,
            num_users,
        )
        key = self._normalize_to_attention_format(
            key,
            "key",
            self.num_kv_heads,
            self.head_size,
            num_users,
        )
        value = self._normalize_to_attention_format(
            value,
            "value",
            self.num_kv_heads,
            self.head_size,
            num_users,
        )

        if key.shape != value.shape:
            raise ValueError(
                f"key and value must match after normalization, got "
                f"key={tuple(key.shape)} value={tuple(value.shape)}"
            )

        users_kv = key.shape[0]
        query_num_tokens = query.shape[1]
        kv_num_tokens = key.shape[1]

        # Determine prefill vs decode mode
        is_prefill = query_num_tokens > 1

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

    def _handle_paged_attention(
        self, inputs, kv_cache: list[torch.Tensor], attn_metadata: TTMetadata
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
            # Prefill: batched across users via N-element batch_idx_tensor.
            key_for_update = inputs.key.transpose(1, 2)
            value_for_update = inputs.value.transpose(1, 2)

            # batch_idx is now built on CPU at setup time (#5154, done upstream)
            # and passed via metadata rather than constructed in-graph per call.
            batch_idxs = attn_metadata.batch_idx
            assert batch_idxs is not None, "batch_idx must be provided for prefill."
            # paged_fill_cache expects batch_idx local to each batch shard, but
            # it's sharded — so % local_batch rebases it to local ids (no-op
            # when dp_size == 1).
            if attn_metadata.dp_size > 1:
                local_batch = key_for_update.shape[0] // attn_metadata.dp_size
                batch_idxs = batch_idxs % local_batch
            k_cache = torch.ops.tt.paged_fill_cache(
                k_cache,
                key_for_update,
                attn_metadata.fill_page_table,
                batch_idx=batch_idxs,
            )
            v_cache = torch.ops.tt.paged_fill_cache(
                v_cache,
                value_for_update,
                attn_metadata.fill_page_table,
                batch_idx=batch_idxs,
            )

        # Preserve tensor identity so XLA reuses the traced graph.
        kv_cache[0].copy_(k_cache)
        kv_cache[1].copy_(v_cache)

    def _compute_full_attention(
        self,
        inputs,
        kv_cache,
        attn_metadata: TTMetadata,
    ) -> torch.Tensor:
        """Compute full attention during the prefill phase.

        Two paths, each a distinct traced graph (no data-dependent control
        flow):

        - Cached-prefix hit (``attn_mask`` set) or KV-sharing layer (no local
          K/V): gather the full per-user K/V slab from the paged cache (no trim
          -- shape fixed at ``num_blocks_per_user * block_size``) and run masked
          SDPA (``is_causal=False``); the mask carries the causal/cached pattern
          (see ``_build_prefill_attn_mask``).
        - Cold prefill (``attn_mask`` None) or no paged cache
          (pooling/profiling): attend ``inputs.key/value`` directly with the
          metadata's ``is_causal``/``attn_mask``. model_runner sets
          ``is_causal = (attn_mask is None)``, so cold prefill gets native
          causal. This skips the redundant paged gather whose full-slab
          read-back degenerated the first token on some models (Llama-3.2-3B).
        """
        has_paged_cache = (
            isinstance(kv_cache, (list, tuple))
            and len(kv_cache) >= 2
            and kv_cache[0].numel() > 0
            and attn_metadata.page_table is not None
        )
        shared_kv_mode = (
            self.kv_sharing_target_layer_name is not None and has_paged_cache
        )
        # Cached-prefix path: a chunk whose prefix is in the paged cache attends
        # over it via chunked_scaled_dot_product_attention (mask + offset internal,
        # no host mask/gather). The model runner sets chunk_start_idx only when
        # chunking occurs and the kernel supports the layout (_chunked_sdpa_active),
        # else it stays None (standard path). The trigger is Python-level, so it
        # traces as a distinct graph -- no data-dependent control flow.
        chunked_prefix = (
            attn_metadata.chunk_start_idx is not None
            and has_paged_cache
            and not shared_kv_mode
        )
        if chunked_prefix:
            chunked_out = torch.ops.tt.chunked_scaled_dot_product_attention(
                inputs.query.transpose(-3, -2),  # [users, n_heads, chunk_len, head]
                kv_cache[0],
                kv_cache[1],
                attn_metadata.page_table,
                attn_metadata.chunk_start_idx,
                scale=self.scale,
            )
            # Back to [users, tokens, num_heads, head_size].
            return chunked_out.transpose(-3, -2)

        # Gather from the paged cache only for a cached-prefix hit (attn_mask
        # set) or a shared-KV layer (no local K/V); cold prefill and the
        # no-paged-cache path attend inputs.key/value directly. See the method
        # docstring for why cold skips the gather.
        must_gather = has_paged_cache and (
            attn_metadata.attn_mask is not None or shared_kv_mode
        )
        if must_gather:
            # Full gather (no trim): shape stays constant across cold/cached
            # prefill so the traced graph is reusable.
            key_for_sdpa = self._gather_paged_to_dense(
                kv_cache[0], attn_metadata.page_table
            )
            value_for_sdpa = self._gather_paged_to_dense(
                kv_cache[1], attn_metadata.page_table
            )
            if attn_metadata.mesh is not None:
                from tt_torch.sharding import sharding_constraint_tensor

                # The paged gather (gather/view/permute/reshape) drops the
                # head-dim sharding the KV cache carries, so under TP the
                # gathered K/V keep full heads while Q stays sharded -> SDPA
                # "Query num heads must be divisible by key/value num heads".
                # Re-assert the head-axis sharding with a graph-emitted
                # sharding_constraint (torch.compile-safe; eager mark_sharding
                # can't be traced inside the fused-prefill fullgraph region).
                key_for_sdpa = sharding_constraint_tensor(
                    key_for_sdpa, attn_metadata.mesh, (None, "model", None, None)
                )
                value_for_sdpa = sharding_constraint_tensor(
                    value_for_sdpa, attn_metadata.mesh, (None, "model", None, None)
                )
            query_for_sdpa = inputs.query.transpose(-3, -2)
            sdpa_kwargs = {
                "is_causal": False,
                "attn_mask": attn_metadata.attn_mask,
                "scale": self.scale,
            }
        else:
            query_for_sdpa = inputs.query.transpose(-3, -2)
            key_for_sdpa = inputs.key.transpose(-3, -2)
            value_for_sdpa = inputs.value.transpose(-3, -2)
            sdpa_kwargs = {
                "is_causal": attn_metadata.is_causal,
                "attn_mask": attn_metadata.attn_mask,
                "scale": self.scale,
            }

        if self.sliding_window is not None:
            sdpa_kwargs["sliding_window_size"] = self.sliding_window

        output = torch.ops.tt.scaled_dot_product_attention(
            query_for_sdpa,
            key_for_sdpa,
            value_for_sdpa,
            **sdpa_kwargs,
        ).transpose(
            -3, -2
        )  # Back to [users, tokens, num_heads, head_size]

        return output

    def _gather_paged_to_dense(
        self,
        cache: torch.Tensor,
        page_table: torch.Tensor,
    ) -> torch.Tensor:
        """Gather a dense K/V tensor from a paged cache buffer.

        Paged layout is ``[num_blocks, num_kv_heads, block_size, head_size]``
        as declared by ``TTAttentionBackend.get_kv_cache_shape``. The
        ``page_table`` gives the block ids per user. We index the blocks,
        re-order dims so the token axis is flattened across blocks, and
        return the full slab ``[users, num_kv_heads, num_blocks * block_size,
        head_size]``. We intentionally do NOT trim to a logical prompt
        length: keeping a constant shape per bucket lets warmup and runtime
        share a single traced graph; the prefill mask masks the padded tail
        (see ``_build_prefill_attn_mask``).

        Uses torch.gather (supported by TT backend) instead of index_select
        (which lowers to ttir.embedding and breaks trace mode).
        """
        num_blocks_per_user = page_table.shape[1]
        num_kv_heads = cache.shape[1]
        block_size = cache.shape[2]
        head_size = cache.shape[3]
        users = page_table.shape[0]

        flat_indices = page_table.reshape(-1).to(torch.int64)
        # Use torch.gather on dim 0 instead of index_select.
        # Expand indices to match cache shape for gather semantics.
        expanded_indices = flat_indices.view(-1, 1, 1, 1).expand(
            -1, num_kv_heads, block_size, head_size
        )
        gathered = torch.gather(cache, 0, expanded_indices)
        # [users * num_blocks_per_user, num_kv_heads, block_size, head_size]
        gathered = gathered.view(
            users, num_blocks_per_user, num_kv_heads, block_size, head_size
        )
        # [users, num_kv_heads, num_blocks_per_user, block_size, head_size]
        gathered = gathered.permute(0, 2, 1, 3, 4).contiguous()
        # [users, num_kv_heads, num_blocks_per_user * block_size, head_size]
        return gathered.reshape(users, num_kv_heads, -1, head_size)

    def _compute_decode_attention(
        self, inputs, kv_cache: list[torch.Tensor], attn_metadata: TTMetadata
    ) -> torch.Tensor:
        """Compute attention for decode phase (paged)."""
        k_cache = kv_cache[0]
        v_cache = kv_cache[1]

        # Adjust for decode kernel expecting query as [1, num_users, num_heads, head]
        # Current query: [users, query_num_tokens, num_heads, head_size]
        # In decode, query_num_tokens == 1 is normal
        query_for_decode = inputs.query.transpose(0, 1)

        decode_kwargs = {
            "cur_pos_tensor": attn_metadata.cache_position,
            "is_causal": attn_metadata.is_causal,
            "attn_mask": attn_metadata.attn_mask,
            "attention_sink": self.sinks,
            "scale": self.scale,
        }
        # Mirror the prefill path: sliding-window layers (e.g. Gemma-4) need
        # the kwarg threaded through, otherwise decode looks at the full KV.
        if self.sliding_window is not None:
            decode_kwargs["sliding_window_size"] = self.sliding_window

        out = torch.ops.tt.paged_scaled_dot_product_attention_decode(
            query_for_decode,
            k_cache,
            v_cache,
            attn_metadata.page_table,
            **decode_kwargs,
        )
        # out: [query_num_tokens, users, num_heads, head_size]
        out = out.transpose(0, 1)  # [users, query_num_tokens, num_heads, head_size]

        return out


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


class TTAttention(Attention):
    """TT wrapper around vLLM Attention with explicit shape handling.

    This class is patched into vLLM at startup and keeps the base Attention
    behavior while normalizing query/key/value inputs to flattened 2D hidden
    representations before dispatch. The output is validated and reshaped back
    to the original query shape.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, **kwargs) -> None:
        super().__init__(
            num_heads=num_heads, head_size=head_size, scale=scale, **kwargs
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        query_shape = query.shape

        def _reshape_to_2d(
            tensor: torch.Tensor | None,
            tensor_name: str,
            expected_heads: int,
            expected_head_size: int,
        ) -> torch.Tensor | None:
            if tensor is None:
                return None

            if tensor.ndim < 2:
                raise ValueError(
                    f"{tensor_name} must have rank >= 2, got shape={tuple(tensor.shape)}"
                )

            expected_hidden = expected_heads * expected_head_size
            if tensor.shape[-1] == expected_hidden:
                return tensor.reshape(-1, expected_hidden)

            if (
                tensor.shape[-2] == expected_heads
                and tensor.shape[-1] == expected_head_size
            ):
                return tensor.reshape(-1, expected_hidden)

            raise ValueError(
                f"{tensor_name} must satisfy one of: "
                f"shape[-1] == {expected_hidden} or "
                f"shape[-2:] == ({expected_heads}, {expected_head_size}); "
                f"got shape={tuple(tensor.shape)}"
            )

        query_2d = _reshape_to_2d(
            query,
            "query",
            self.num_heads,
            self.head_size,
        )
        key_2d = _reshape_to_2d(
            key,
            "key",
            self.num_kv_heads,
            self.head_size,
        )
        value_2d = _reshape_to_2d(
            value,
            "value",
            self.num_kv_heads,
            self.head_size_v,
        )

        output_2d = super().forward(query_2d, key_2d, value_2d, output_shape)

        if output_2d.numel() != query.numel():
            raise ValueError(
                "Cannot reshape attention output back to query shape: "
                f"output_shape={tuple(output_2d.shape)}, "
                f"query_shape={tuple(query_shape)}"
            )

        return output_2d.reshape(query_shape)


class TTEncoderOnlyAttention(TTAttention):
    """Encoder-only TT attention wrapper with explicit backend selection.

    This variant builds an encoder-only backend via
    ``create_encoder_only_attention_backend(get_attn_backend(...))`` and
    enforces ``AttentionType.ENCODER_ONLY`` semantics. It inherits TTAttention
    shape handling and declares no KV cache support.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        cache_config: CacheConfig | None = None,
        attn_type: str | None = None,
        **kwargs,
    ) -> None:
        dtype = torch.get_default_dtype()
        kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        underlying_attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            attn_type=AttentionType.ENCODER_ONLY,
        )
        attn_backend = create_encoder_only_attention_backend(underlying_attn_backend)
        if attn_type is not None:
            assert (
                attn_type == AttentionType.ENCODER_ONLY
            ), "TTEncoderOnlyAttention only supports AttentionType.ENCODER_ONLY"
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=attn_backend,
            attn_type=AttentionType.ENCODER_ONLY,
            **kwargs,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return None
