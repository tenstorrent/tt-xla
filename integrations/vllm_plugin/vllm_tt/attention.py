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
from vllm.logger import init_logger
from vllm.utils import cdiv, next_power_of_2

logger = init_logger(__name__)

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
        batch_size: int, num_heads: int, max_seq_len: int, head_size: int
    ) -> tuple[int, ...]:
        # [2, batch_size, num_heads, max_cache_len), head_dim]
        padded_head_size = (
            cdiv(head_size, TT_HEAD_SIZE_ALIGNMENT) * TT_HEAD_SIZE_ALIGNMENT
        )
        return (2, batch_size, num_heads, max_seq_len, head_size)

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
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Used in the TTAttentionBackendImpl
    context_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_seqs: torch.Tensor
    # padding_in_inputs: torch.Tensor
    attn_mask: torch.Tensor
    is_causal: bool


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

        if attn_type == "encoder_only":
            attn_type = AttentionType.ENCODER_ONLY

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
        is_batched = query.ndim > 2
        query_hidden_size = query.shape[-1]
        query_num_tokens = query.shape[-2]
        query_batch_size = query.shape[0] if is_batched else 1

        kv_hidden_size = key.shape[-1]
        kv_num_tokens = key.shape[-2]
        kv_batch_size = key.shape[0] if is_batched else 1

        query = query.reshape(
            query_batch_size,
            query_num_tokens,
            query_hidden_size // self.head_size,
            self.head_size,
        ).transpose(
            -3, -2
        )  # [batch, num_tokens, num_heads, head_size]
        key = key.reshape(
            kv_batch_size,
            kv_num_tokens,
            kv_hidden_size // self.head_size,
            self.head_size,
        ).transpose(
            -3, -2
        )  # [batch, num_tokens, num_kv_heads, head_size]
        value = value.reshape(
            kv_batch_size,
            kv_num_tokens,
            kv_hidden_size // self.head_size,
            self.head_size,
        ).transpose(
            -3, -2
        )  # [batch, num_tokens, num_kv_heads, head_size]

        if kv_cache.numel() > 1:
            cache_position = (attn_metadata.context_lens[:1] - 1).to(query.device)

            k_cache = kv_cache[0]
            v_cache = kv_cache[1]

            if query.shape[-2] == 1:
                k_cache = torch.ops.tt.update_cache(k_cache, key, cache_position)
                v_cache = torch.ops.tt.update_cache(v_cache, value, cache_position)
            else:
                # See the comment in this function for more details.
                k_cache = fill_cache_workaround(k_cache.shape, key)
                v_cache = fill_cache_workaround(v_cache.shape, value)
            new_kv_cache = torch.stack([k_cache, v_cache], dim=0)
            key = k_cache
            value = v_cache
            kv_cache.copy_(new_kv_cache)

        if query.shape[-2] == 1:
            query = query.reshape(1, query.shape[0], query.shape[1], query.shape[3])
            cur_pos_tensor = attn_metadata.context_lens[:1].to(query.device)
            out = torch.ops.tt.scaled_dot_product_attention_decode(
                query,
                key,
                value,
                cur_pos_tensor,
                is_causal=attn_metadata.is_causal,
                attn_mask=attn_metadata.attn_mask,
            )
            out = out.transpose(-3, -2)
            out = out.reshape(query_num_tokens, query_hidden_size)
            return out
        else:
            output = torch.ops.tt.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=attn_metadata.is_causal,
                attn_mask=attn_metadata.attn_mask,
            ).transpose(-3, -2)
            if is_batched:
                return output.reshape(
                    query_batch_size, query_num_tokens, query_hidden_size
                )
            else:
                return output.reshape(query_num_tokens, query_hidden_size)


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
