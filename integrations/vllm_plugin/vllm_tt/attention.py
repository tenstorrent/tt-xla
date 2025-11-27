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
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import VllmConfig
from vllm.utils import cdiv, next_power_of_2

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


class TTAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TT_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["TTAttentionBackendImpl"]:
        return TTAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> type["TTMetadata"]:
        return TTMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
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


import pdb


@dataclass
class TTMetadata:
    # Used in the TTAttentionBackendImpl
    cache_position: torch.Tensor
    attn_mask: torch.Tensor
    page_table: torch.Tensor
    is_causal: bool

    def __init__(
        self,
        cache_position: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        page_table: torch.Tensor = None,
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

    # @torch.compiler.disable
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
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

        # === Common metadata ===
        num_users = attn_metadata.cache_position.shape[0]  # logical batch size
        orig_query_shape = query.shape
        orig_query_ndim = query.ndim

        # ----- Reshape query to [users, tokens_per_user, hidden] -----
        if orig_query_ndim == 3:
            # [batch, seq, hidden]
            users = query.shape[0]
            query_num_tokens = query.shape[1]
            hidden_size = query.shape[2]
            assert users == num_users, (
                f"query batch dim ({users}) and cache_position num_users "
                f"({num_users}) mismatch."
            )
            query = query  # Already [users, query_num_tokens, hidden]
        elif orig_query_ndim == 2:
            # [total_tokens, hidden] (vLLM style)
            total_tokens = query.shape[0]
            hidden_size = query.shape[1]
            users = num_users
            assert (
                total_tokens % users == 0
            ), f"total_tokens ({total_tokens}) not divisible by num_users ({users})."
            query_num_tokens = total_tokens // users  # tokens per user
            query = query.view(users, query_num_tokens, hidden_size)
        else:
            raise ValueError(f"Unsupported query rank: {orig_query_ndim}")

        # prefill vs decode: determined by number of tokens per user
        is_prefill = query_num_tokens > 1

        # ----- Reshape key/value to [users_kv, kv_num_tokens, hidden] -----
        # key
        if key.ndim == 3:
            users_kv = key.shape[0]
            kv_num_tokens = key.shape[1]
            kv_hidden_size = key.shape[2]
            # Usually users_kv should equal num_users
        elif key.ndim == 2:
            total_k_tokens = key.shape[0]
            kv_hidden_size = key.shape[1]
            users_kv = users  # Assume same users as query
            assert (
                total_k_tokens % users_kv == 0
            ), f"total_k_tokens ({total_k_tokens}) not divisible by users_kv ({users_kv})."
            kv_num_tokens = total_k_tokens // users_kv
            key = key.view(users_kv, kv_num_tokens, kv_hidden_size)
        else:
            raise ValueError(f"Unsupported key rank: {key.ndim}")

        # value
        if value.ndim == 3:
            users_v = value.shape[0]
            v_num_tokens = value.shape[1]
            v_hidden_size = value.shape[2]
            assert (
                users_v == users_kv and v_num_tokens == kv_num_tokens
            ), "key/value shape mismatch."
        elif value.ndim == 2:
            total_v_tokens = value.shape[0]
            v_hidden_size = value.shape[1]
            users_v = users_kv
            assert (
                total_v_tokens % users_v == 0
            ), f"total_v_tokens ({total_v_tokens}) not divisible by users_v ({users_v})."
            v_num_tokens = total_v_tokens // users_v
            assert v_num_tokens == kv_num_tokens, "key/value token count mismatch."
            value = value.view(users_v, v_num_tokens, v_hidden_size)
        else:
            raise ValueError(f"Unsupported value rank: {value.ndim}")

        # === Reshape Q/K/V to [batch(users), tokens, num_heads, head_size] ===
        num_heads = hidden_size // self.head_size
        assert hidden_size % self.head_size == 0

        query = query.reshape(users, query_num_tokens, num_heads, self.head_size)
        key = key.reshape(
            users_kv, kv_num_tokens, kv_hidden_size // self.head_size, self.head_size
        )
        value = value.reshape(
            users_kv, kv_num_tokens, v_hidden_size // self.head_size, self.head_size
        )

        if kv_cache.numel() > 1:
            # === paged attention path ===
            # When kv_cache exists
            k_cache = kv_cache[0]
            v_cache = kv_cache[1]

            if not is_prefill:
                key_for_update = key.transpose(
                    0, 1
                )  # [users, num_heads, tokens, head_size]
                value_for_update = value.transpose(0, 1)

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
                # prefill: filling multiple tokens at once
                # paged_fill_cache assumes it receives [users, tokens, num_heads, head_size] directly
                key_for_update = key.transpose(
                    1, 2
                )  # [users, num_heads, tokens, head_size]
                value_for_update = value.transpose(1, 2)
                for batch_idx in range(users):
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

            new_kv_cache = torch.stack([k_cache, v_cache], dim=0)
            kv_cache.copy_(new_kv_cache)

        if is_prefill:
            # non-paged path
            # scaled_dot_product_attention expects [B, N_tokens, N_heads, H]
            query_for_sdpa = query.transpose(
                -3, -2
            )  # [users, num_heads, tokens, head_size]
            key_for_sdpa = key.transpose(-3, -2)
            value_for_sdpa = value.transpose(-3, -2)

            output = torch.ops.tt.scaled_dot_product_attention(
                query_for_sdpa,
                key_for_sdpa,
                value_for_sdpa,
                is_causal=attn_metadata.is_causal,
                attn_mask=attn_metadata.attn_mask,
            ).transpose(
                -3, -2
            )  # Back to [users, tokens, num_heads, head_size]

            # output = output.reshape(
            #     users, query_num_tokens, -1
            # )  # Flatten to hidden dim

            # Return to original query rank
            if orig_query_ndim == 3:
                # [batch, seq, hidden]
                return output
            else:
                # [total_tokens, hidden]
                total_tokens = users * query_num_tokens
                return output.reshape(total_tokens, -1)

        # ---- paged decode ----
        # Adjust for decode kernel expecting query as [1, num_users, num_heads, head]
        # Current query: [users, query_num_tokens, num_heads, head_size]
        # In decode, query_num_tokens == 1 is normal
        query_for_decode = query.transpose(
            0, 1
        )  # [query_num_tokens, users, num_heads, head_size]

        out = torch.ops.tt.paged_scaled_dot_product_attention_decode(
            query_for_decode,
            k_cache,
            v_cache,
            attn_metadata.page_table,
            cur_pos_tensor=attn_metadata.cache_position,
            is_causal=attn_metadata.is_causal,
            attn_mask=attn_metadata.attn_mask,
        )
        # out: assumed to be [query_num_tokens, users, num_heads, head_size]
        out = out.transpose(-3, -2)  # [query_num_tokens, users, head_size, num_heads]
        out = out.transpose(0, 1)  # [users, query_num_tokens, head_size, num_heads]
        out = out.reshape(
            users, query_num_tokens, -1
        )  # [users, query_num_tokens, hidden]

        if orig_query_ndim == 3:
            # Original [batch, seq, hidden]
            return out
        else:
            # Original [total_tokens, hidden]
            total_tokens = users * query_num_tokens
            return out.reshape(total_tokens, -1)


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
