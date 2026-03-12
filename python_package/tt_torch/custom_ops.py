# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Optional, Tuple, Union

import torch
from torch_xla.experimental import stablehlo_custom_call

_perf_logger = logging.getLogger("tt_torch.custom_ops.perf")
_perf_accum = {}


def _perf_log(name, elapsed):
    """Accumulate and periodically log timing for custom op CPU paths."""
    if name not in _perf_accum:
        _perf_accum[name] = {"count": 0, "total": 0.0}
    _perf_accum[name]["count"] += 1
    _perf_accum[name]["total"] += elapsed
    import sys

    print(
        f"[PERF] {name}: {elapsed:.3f}s (call #{_perf_accum[name]['count']}, "
        f"total {_perf_accum[name]['total']:.3f}s)",
        file=sys.stderr,
        flush=True,
    )


@torch.library.custom_op(
    "tt::mark_argument_attributes", mutates_args=[], device_types=["cpu", "xla"]
)
def mark_argument_attributes(
    tensor: torch.Tensor, argument_type: str, name: str = None
) -> torch.Tensor:
    """
    This function is a custom registered operator accessible as torch.ops.tt.mark_argument_attributes.
    You may only apply this function to a tensor which is on an XLA device.
    This function will annotate the tensor in a compiled program with a "name" and "argument_type" attribute.
    """
    if tensor.device.type == "cpu":
        return tensor.clone()

    assert isinstance(
        argument_type, str
    ), f"argument_type must be a string, received {type(argument_type)}"
    assert argument_type in [
        "input",
        "parameter",
        "constant",
    ], f"argument_type must be one of 'input', 'parameter', or 'constant', received {argument_type}"

    frontend_attributes = {"ttcore.argument_type": argument_type}
    if name is not None:
        frontend_attributes["ttir.name"] = name

    # @LPanosTT: stablehlo_custom_call causes issues (sometimes) within XLA for shapes which are 2D (or less?), it is unclear why.
    # There is a todo within torch-xla addressing this: venv/lib/python3.10/site-packages/torch_xla/experimental/stablehlo_custom_call.py
    # I have implemented a workaround for this by reshaping the tensor to 2D if it is less than 2D, then reshaping back to the original shape.
    # This should not have performance impact as the custom call below will be removed by the graph, and the reshapes will thus be placed back-to-back. tt-mlir will fold both of them out.
    original_shape = list(tensor.shape)
    if len(tensor.shape) < 3:
        extra_dims = [1] * (3 - len(original_shape))
        tensor = tensor.reshape((*extra_dims, *original_shape))
    result = stablehlo_custom_call.stablehlo_custom_call(
        [tensor],
        "tt.mark_argument",
        [tensor.shape],
        [tensor.dtype],
        frontend_attributes=frontend_attributes,
    )
    if len(original_shape) < 3:
        result = result.reshape(original_shape)
    return result


@mark_argument_attributes.register_fake
def _(tensor: torch.Tensor, argument_type: str, name: str = None) -> torch.Tensor:
    """
    FakeTensor implementation of torch.ops.tt.mark_argument_attributes.
    This must be implemented in order for dynamo to trace the function.
    returns:
        - tensor: the same tensor that was passed in
    """
    return tensor.clone()


@mark_argument_attributes.register_autograd
def _(ctx, grad_output):
    """
    Autograd implementation for mark_argument_attributes.
    This op is just metadata annotation, so gradients pass through unchanged.
    Returns gradients for: (tensor, argument_type, name)
    """
    return grad_output, None, None


@torch.library.custom_op(
    "tt::sharding_constraint", mutates_args=[], device_types=["cpu", "xla"]
)
def sharding_constraint(tensor: torch.Tensor, sdy_sharding: str) -> torch.Tensor:
    """
    Apply a sharding constraint to a tensor for Shardy propagation.

    This function is a custom registered operator accessible as torch.ops.tt.sharding_constraint.
    It creates a stablehlo.custom_call @tt.sharding_constraint op that tt-mlir converts to sdy.sharding_constraint.

    Args:
        tensor: The input tensor to apply sharding to
        sdy_sharding: The sdy.sharding string (e.g., '#sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {}, {}]>]>')

    Returns:
        A tensor with sharding constraint applied
    """
    if tensor.device.type == "cpu":
        return tensor.clone()

    frontend_attributes = {
        "xla.sdy.sharding": sdy_sharding,
    }

    result = stablehlo_custom_call.stablehlo_custom_call(
        [tensor],
        "tt.sharding_constraint",  # tt-mlir converts this to sdy.sharding_constraint
        [tensor.shape],
        [tensor.dtype],
        frontend_attributes=frontend_attributes,
    )

    return result


@sharding_constraint.register_fake
def _(tensor: torch.Tensor, sdy_sharding: str) -> torch.Tensor:
    """
    FakeTensor implementation of torch.ops.tt.sharding_constraint.
    This must be implemented in order for dynamo to trace the function.
    """
    return tensor.clone()


@sharding_constraint.register_autograd
def _(ctx, grad_output):
    """
    Autograd implementation for sharding_constraint.
    This op only applies sharding metadata, so gradients pass through unchanged.
    Returns gradients for: (tensor, sdy_sharding)
    """
    return grad_output, None


@torch.library.custom_op(
    "tt::scaled_dot_product_attention", mutates_args=[], device_types=["xla", "cpu"]
)
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    is_causal: bool = True,
    scale: float = None,
) -> torch.Tensor:

    assert (
        len(query.shape) == 4
    ), "query must be a 4D tensor: [B, num_heads, query_seq_len, head_size]."
    assert (
        len(key.shape) == 4
    ), "key must be a 4D tensor: [B, num_kv_heads, kv_seq_len, head_size]."
    assert (
        len(value.shape) == 4
    ), "value must be a 4D tensor: [B, num_kv_heads, kv_seq_len, head_size]."

    assert key.shape == value.shape, "key and value must have the same shape."
    assert (
        key.shape[-1] == query.shape[-1]
    ), "key and query must have the same head size."

    assert (
        query.shape[1] % key.shape[1] == 0
    ), "num_heads must be divisible by num_kv_heads."

    # The CPU implementation of this op will funtion correctly if this invariant is not satisfied.
    # However, this custom op is intended to exactly replicate the behavior of the ttnn op, so we will enforce this invariant.
    assert (
        query.shape[2] % 32 == 0
    ), f"query sequence length must be divisible by 32 but got {query.shape[2]}."

    # assert query.shape[0] == 1, "query must have dim 0 equal to 1."
    assert (
        query.shape[0] == key.shape[0]
    ), "query and key must have the same batch size."

    assert (
        query.device == key.device == value.device
    ), "query, key, and value must be on the same device."
    if attn_mask is not None:
        assert (
            attn_mask.device == query.device
        ), "attn_mask must be on the same device as query, key, and value."

        assert (
            is_causal == False
        ), "is_causal attribute can't be True if attn_mask is available."

        assert (
            query.shape[0] == attn_mask.shape[0]
        ), "Attention mask batch size must match query batch size."

    if query.device.type == "xla":
        inputs = [query, key, value]
        if attn_mask is not None:
            inputs.append(attn_mask)

        frontend_attributes = {"is_causal": str(is_causal)}
        if scale is not None:
            frontend_attributes["scale"] = str(scale)

        return stablehlo_custom_call.stablehlo_custom_call(
            inputs,
            "tt.scaled_dot_product_attention",
            [query.shape],
            [query.dtype],
            frontend_attributes=frontend_attributes,
        )

    elif query.device.type == "cpu":
        # Enable GQA as the ttnn op handles GQA automatically.
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=True,
        )
    else:
        raise ValueError(f"Unsupported device type: {query.device.type}")


@scaled_dot_product_attention.register_fake
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    is_causal: bool = True,
    scale: float = None,
) -> torch.Tensor:
    return torch.zeros_like(query)


@torch.library.custom_op(
    "tt::scaled_dot_product_attention_decode",
    mutates_args=[],
    device_types=["xla", "cpu"],
)
def scaled_dot_product_attention_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cur_pos_tensor: torch.Tensor,
    attn_mask: torch.Tensor = None,
    attention_sink: torch.Tensor = None,
    is_causal: bool = True,
    scale: float = None,
) -> torch.Tensor:

    assert (
        len(query.shape) == 4
    ), "query must be a 4D tensor: [1, B, num_heads, head_size]."
    assert (
        len(key.shape) == 4
    ), "key must be a 4D tensor: [B, num_kv_heads, seq_len, head_size]."
    assert (
        len(value.shape) == 4
    ), "value must be a 4D tensor: [B, num_kv_heads, seq_len, head_size]."
    assert len(cur_pos_tensor.shape) == 1, "cur_pos_tensor must be a 1D tensor: [B]."

    assert key.shape == value.shape, "key and value must have the same shape."
    assert (
        key.shape[-1] == query.shape[-1]
    ), "key and query must have the same head size."

    assert (
        query.shape[2] % key.shape[1] == 0
    ), f"num_heads must be divisible by num_kv_heads. Query shape: {query.shape}, key shape: {key.shape}."

    assert query.shape[0] == 1, "query must have dim 0 equal to 1."
    assert (
        query.shape[1] == key.shape[0]
    ), "query and key must have the same batch size."

    if is_causal:
        assert attn_mask is None, "attn_mask must be None when is_causal is True."

    if query.device.type == "xla":

        inputs = [query, key, value, cur_pos_tensor]
        if attn_mask is not None:
            inputs.append(attn_mask)
        if attention_sink is not None:
            inputs.append(attention_sink)

        frontend_attributes = {
            "is_causal": str(is_causal),
            "has_attention_mask": str(attn_mask is not None),
            "has_attention_sink": str(attention_sink is not None),
        }
        if scale is not None:
            frontend_attributes["scale"] = str(scale)

        return stablehlo_custom_call.stablehlo_custom_call(
            inputs,
            "tt.scaled_dot_product_attention_decode",
            [query.shape],
            [query.dtype],
            frontend_attributes=frontend_attributes,
        )

    elif query.device.type == "cpu":
        # TODO(@LPanosTT): Model the behavior of the op when an attention_sink is provided.
        batch_size = query.shape[1]
        num_heads = query.shape[2]
        head_size = query.shape[3]
        max_seq_len = key.shape[-2]
        query = query.reshape(batch_size, num_heads, 1, head_size)
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(batch_size, num_heads, 1, max_seq_len)
        else:
            # For ttnn.scaled_dot_product_attention_decode, is_causal indicates that the attention should
            # disregard tokens to the right of the current position in the KV cache. In PyTorch
            # scaled_dot_product_attention, is_causal=True creates a triangular mask of shape
            # (query_seq_len, max_seq_len). Since query_seq_len is 1 for the decode op, this produces a
            # single mask row that is all -inf except for the first element (0), which mismatches the
            # ttnn op’s behavior. We therefore construct an additive mask that replicates ttnn semantics.
            attn_mask = torch.zeros(
                batch_size, num_heads, 1, max_seq_len, dtype=query.dtype
            )
            # For each user (batch), mask out tokens to the right of the current position in the KV cache.
            for batch_idx in range(batch_size):
                attn_mask[batch_idx, ..., cur_pos_tensor[batch_idx] + 1 :] = float(
                    "-inf"
                )

        # Enable GQA as the ttnn op handles GQA automatically.
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask, is_causal=False, scale=scale, enable_gqa=True
        ).reshape(1, batch_size, num_heads, head_size)
    else:
        raise ValueError(f"Unsupported device type: {query.device.type}")


@scaled_dot_product_attention_decode.register_fake
def scaled_dot_product_attention_decode_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cur_pos_tensor: torch.Tensor,
    attn_mask: torch.Tensor = None,
    attention_sink: torch.Tensor = None,
    is_causal: bool = True,
    scale: float = None,
) -> torch.Tensor:
    return torch.zeros_like(query)


@torch.library.custom_op(
    "tt::update_cache", mutates_args=[], device_types=["xla", "cpu"]
)
def update_cache(
    cache: torch.Tensor,
    fill_value: torch.Tensor,
    cache_position: torch.Tensor,
    batch_offset: int = None,
) -> torch.Tensor:
    assert (
        len(cache.shape) == 4
    ), "cache must be a 4D tensor: [B, num_heads, max_seq_len, head_size]."
    assert (
        len(fill_value.shape) == 4
    ), "fill_value must be a 4D tensor: [B, num_heads, 1, head_size]."
    assert (
        fill_value.shape[-2] == 1
    ), "fill_value must have dim -2 equal to 1 as the cache cannot be updated with more than one token at a time."
    assert (
        batch_offset is not None or cache.shape[0] == 1
    ), "batch_offset must be provided if the batch size is not 1."
    assert cache_position.shape == (1,), "cache_position must be a 1D tensor."

    if batch_offset is None:
        batch_offset = 0

    assert batch_offset == 0, "Only batch_offset == 0 is supported for currently."
    if cache.device.type == "cpu":
        cache = cache.clone()
        cache[:, :, cache_position, :] = fill_value
        return cache
    else:
        return stablehlo_custom_call.stablehlo_custom_call(
            [cache, fill_value, cache_position],
            "tt.update_cache",
            [
                cache.shape,
            ],
            [
                cache.dtype,
            ],
            frontend_attributes={"batch_offset": str(batch_offset)},
        )


@update_cache.register_fake
def update_cache_fake(
    cache: torch.Tensor,
    fill_value: torch.Tensor,
    cache_position: torch.Tensor,
    batch_offset: int = None,
) -> torch.Tensor:
    return torch.zeros_like(cache)


@torch.library.custom_op("tt::fill_cache", mutates_args=[], device_types=["xla", "cpu"])
def fill_cache(
    cache: torch.Tensor, fill_value: torch.Tensor, batch_offset: int = None
) -> torch.Tensor:
    assert (
        len(cache.shape) == 4
    ), "cache must be a 4D tensor: [B, num_heads, max_seq_len, head_size]."
    assert (
        len(fill_value.shape) == 4
    ), "fill_value must be a 4D tensor: [B, num_heads, seq_len, head_size]."
    assert (
        fill_value.shape[-2] <= cache.shape[-2]
    ), f"fill_value must have dim -2 less than or equal to cache.shape[-2] as the cache cannot be filled with more tokens than the cache can hold. Recieved fill_value.shape = {fill_value.shape}, cache.shape = {cache.shape}."
    assert (
        batch_offset is not None or cache.shape[0] == 1
    ), "batch_offset must be provided if the batch size is not 1."

    if batch_offset is None:
        batch_offset = 0

    assert batch_offset == 0, "Only batch_offset == 0 is supported for currently."
    if cache.device.type == "cpu":
        cache = cache.clone()
        cache[:, :, : fill_value.shape[-2], :] = fill_value
        return cache
    else:

        return stablehlo_custom_call.stablehlo_custom_call(
            [cache, fill_value],
            "tt.fill_cache",
            [
                cache.shape,
            ],
            [
                cache.dtype,
            ],
            frontend_attributes={"batch_offset": str(batch_offset)},
        )


@fill_cache.register_fake
def fill_cache_fake(
    cache: torch.Tensor, fill_value: torch.Tensor, batch_offset: int = None
) -> torch.Tensor:
    return torch.zeros_like(cache)


@torch.library.custom_op(
    "tt::paged_update_cache", mutates_args=[], device_types=["xla", "cpu"]
)
def paged_update_cache(
    cache: torch.Tensor,
    fill_value: torch.Tensor,
    update_indices: torch.Tensor,
    page_table: torch.Tensor,
    share_cache: bool = False,
) -> torch.Tensor:
    device = cache.device
    if device.type == "xla":
        return stablehlo_custom_call.stablehlo_custom_call(
            [cache, fill_value, update_indices, page_table],
            "tt.paged_update_cache",
            [cache.shape],
            [cache.dtype],
            frontend_attributes={"share_cache": str(share_cache)},
        )
    elif device.type == "cpu":
        cache = cache.clone()
        num_users = update_indices.shape[0]
        block_size = cache.shape[-2]
        num_heads = cache.shape[-3]

        # Find which block (per user) are being updated
        block_indices = update_indices // block_size

        # Find how deep into the block the update is
        block_offsets = update_indices % block_size

        user_range = torch.arange(num_users)

        fill_values_view = fill_value[0, :, :num_heads, :]
        cache[page_table[user_range, block_indices], :, block_offsets, :] = (
            fill_values_view
        )

        return cache
    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@paged_update_cache.register_fake
def paged_update_cache_fake(
    cache: torch.Tensor,
    fill_value: torch.Tensor,
    update_indices: torch.Tensor,
    page_table: torch.Tensor,
    share_cache=False,
) -> torch.Tensor:
    return torch.zeros_like(cache)


@torch.library.custom_op(
    "tt::paged_fill_cache", mutates_args=[], device_types=["xla", "cpu"]
)
def paged_fill_cache(
    cache: torch.Tensor,
    fill_value: torch.Tensor,
    page_table: torch.Tensor,
    batch_idx: torch.Tensor = None,
) -> torch.Tensor:
    device = cache.device
    if batch_idx is None:
        batch_idx = torch.tensor([0], dtype=torch.int32, device=device)
    if device.type == "xla":
        inputs = [cache, fill_value, page_table, batch_idx]
        return stablehlo_custom_call.stablehlo_custom_call(
            inputs,
            "tt.paged_fill_cache",
            [cache.shape],
            [cache.dtype],
        )

    elif device.type == "cpu":
        cache = cache.clone()

        block_size = cache.shape[-2]
        fill_seq_len = fill_value.shape[-2]
        num_heads = fill_value.shape[-3]

        batch_block_indices = page_table[batch_idx.item()]

        num_blocks_to_fill = fill_seq_len // block_size
        part_of_final_block_to_fill = fill_seq_len % block_size

        if num_blocks_to_fill > 0:
            fill_value_in_blocks_shape = [
                num_blocks_to_fill,
                num_heads,
                block_size,
                fill_value.shape[-1],
            ]
            if part_of_final_block_to_fill > 0:
                fill_value_first_blocks = fill_value[
                    :, :, :-part_of_final_block_to_fill, :
                ]
                cache[batch_block_indices[:num_blocks_to_fill]] = (
                    fill_value_first_blocks.reshape(
                        1,
                        num_heads,
                        num_blocks_to_fill,
                        block_size,
                        fill_value.shape[-1],
                    )
                    .transpose(0, 2)
                    .reshape(fill_value_in_blocks_shape)
                )
            else:
                cache[batch_block_indices[:num_blocks_to_fill]] = (
                    fill_value.reshape(
                        1,
                        num_heads,
                        num_blocks_to_fill,
                        block_size,
                        fill_value.shape[-1],
                    )
                    .transpose(0, 2)
                    .reshape(fill_value_in_blocks_shape)
                )

        if part_of_final_block_to_fill > 0:
            cache[
                batch_block_indices[num_blocks_to_fill : num_blocks_to_fill + 1],
                :,
                :part_of_final_block_to_fill,
                :,
            ] = fill_value[:, :, -part_of_final_block_to_fill:, :]

        return cache
    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@paged_fill_cache.register_fake
def paged_fill_cache_fake(
    cache: torch.Tensor,
    fill_value: torch.Tensor,
    page_table: torch.Tensor,
    batch_idx: torch.Tensor = None,
) -> torch.Tensor:
    return torch.zeros_like(cache)


@torch.library.custom_op(
    "tt::paged_fill_cache", mutates_args=[], device_types=["xla", "cpu"]
)
def paged_fill_cache(
    cache: torch.Tensor,
    fill_value: torch.Tensor,
    page_table: torch.Tensor,
    batch_idx: torch.Tensor = None,
) -> torch.Tensor:
    device = cache.device
    if batch_idx is None:
        batch_idx = torch.tensor([0], dtype=torch.int32, device=device)
    if device.type == "xla":
        inputs = [cache, fill_value, page_table, batch_idx]
        return stablehlo_custom_call.stablehlo_custom_call(
            inputs,
            "tt.paged_fill_cache",
            [cache.shape],
            [cache.dtype],
        )

    elif device.type == "cpu":
        cache = cache.clone()

        block_size = cache.shape[-2]
        fill_seq_len = fill_value.shape[-2]
        num_heads = fill_value.shape[-3]

        batch_block_indices = page_table[batch_idx.item()]

        num_blocks_to_fill = fill_seq_len // block_size
        part_of_final_block_to_fill = fill_seq_len % block_size

        if num_blocks_to_fill > 0:
            fill_value_in_blocks_shape = [
                num_blocks_to_fill,
                num_heads,
                block_size,
                fill_value.shape[-1],
            ]
            if part_of_final_block_to_fill > 0:
                fill_value_first_blocks = fill_value[
                    :, :, :-part_of_final_block_to_fill, :
                ]
                cache[batch_block_indices[:num_blocks_to_fill]] = (
                    fill_value_first_blocks.reshape(
                        1,
                        num_heads,
                        num_blocks_to_fill,
                        block_size,
                        fill_value.shape[-1],
                    )
                    .transpose(0, 2)
                    .reshape(fill_value_in_blocks_shape)
                )
            else:
                cache[batch_block_indices[:num_blocks_to_fill]] = (
                    fill_value.reshape(
                        1,
                        num_heads,
                        num_blocks_to_fill,
                        block_size,
                        fill_value.shape[-1],
                    )
                    .transpose(0, 2)
                    .reshape(fill_value_in_blocks_shape)
                )

        if part_of_final_block_to_fill > 0:
            cache[
                batch_block_indices[num_blocks_to_fill : num_blocks_to_fill + 1],
                :,
                :part_of_final_block_to_fill,
                :,
            ] = fill_value[:, :, -part_of_final_block_to_fill:, :]

        return cache
    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@paged_fill_cache.register_fake
def paged_fill_cache_fake(
    cache: torch.Tensor,
    fill_value: torch.Tensor,
    page_table: torch.Tensor,
    batch_idx: torch.Tensor = None,
) -> torch.Tensor:
    return torch.zeros_like(cache)


@torch.library.custom_op(
    "tt::paged_scaled_dot_product_attention_decode",
    mutates_args=[],
    device_types=["xla", "cpu"],
)
def paged_scaled_dot_product_attention_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    page_table: torch.Tensor,
    is_causal: bool = False,
    attn_mask: torch.Tensor = None,
    cur_pos_tensor: torch.Tensor = None,
    attention_sink: torch.Tensor = None,
    scale: float = None,
) -> torch.Tensor:
    device = query.device

    if is_causal:
        assert attn_mask is None, "attention_mask must be None when is_causal is True."
        assert (
            cur_pos_tensor is not None
        ), "cur_pos_tensor must be provided when is_causal is True."
    if attn_mask is not None:
        assert not is_causal, "attention_mask must be None when is_causal is True."

    if device.type == "xla":
        attrs = {
            "has_attention_mask": "False",
            "has_cur_pos_tensor": "False",
            "has_attention_sink": "False",
            "is_causal": str(is_causal),
        }

        if scale is not None:
            attrs["scale"] = str(scale)

        inputs = [query, key, value, page_table]
        if attn_mask is not None:
            attrs["has_attention_mask"] = "True"
            inputs.append(attn_mask)
        if cur_pos_tensor is not None:
            attrs["has_cur_pos_tensor"] = "True"
            inputs.append(cur_pos_tensor)
        if attention_sink is not None:
            attrs["has_attention_sink"] = "True"
            inputs.append(attention_sink)

        return stablehlo_custom_call.stablehlo_custom_call(
            inputs,
            "tt.paged_scaled_dot_product_attention_decode",
            [query.shape],
            [query.dtype],
            frontend_attributes=attrs,
        )
    elif device.type == "cpu":
        # Select the proper key and value blocks based on the page table
        block_size = key.shape[-2]
        num_heads = key.shape[-3]
        num_users = cur_pos_tensor.shape[0]
        head_size = key.shape[-1]

        num_blocks_per_user = page_table.shape[1]
        max_seq_len = num_blocks_per_user * block_size
        causal_mask = torch.zeros(num_users, max_seq_len, dtype=query.dtype)

        new_key = torch.zeros(
            num_users, num_heads, max_seq_len, head_size, dtype=query.dtype
        )
        new_value = torch.zeros(
            num_users, num_heads, max_seq_len, head_size, dtype=query.dtype
        )

        for i in range(num_users):
            block_indices = page_table[i]

            user_key_blocks = key[block_indices]
            user_value_blocks = value[block_indices]

            # Flatten blocks into seq len
            user_key = user_key_blocks.transpose(0, 1)  # Move head dim to the front
            user_value = user_value_blocks.transpose(0, 1)  # Move head dim to the front

            # Select the proper key and value blocks based on the current position
            user_key = user_key.reshape(
                num_heads, block_size * num_blocks_per_user, head_size
            )
            user_value = user_value.reshape(
                num_heads, block_size * num_blocks_per_user, head_size
            )

            new_key[i] = user_key
            new_value[i] = user_value

            causal_mask[i, cur_pos_tensor[i] + 1 :] = float("-inf")

        query = query.reshape(num_users, num_heads, 1, head_size)

        attn_mask = (
            causal_mask.reshape(num_users, 1, 1, max_seq_len)
            if is_causal
            else attn_mask
        )

        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
        scale = 1 / head_size**0.5 if scale is None else scale
        attn_weight = query @ new_key.transpose(-2, -1) * scale
        attn_weight += attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        out = attn_weight @ new_value
        return out.reshape(1, num_users, num_heads, head_size)

    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@paged_scaled_dot_product_attention_decode.register_fake
def paged_scaled_dot_product_attention_decode_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    page_table: torch.Tensor,
    is_causal: bool = False,
    attention_mask: torch.Tensor = None,
    cur_pos_tensor: torch.Tensor = None,
    attention_sink: torch.Tensor = None,
    scale: float = None,
) -> torch.Tensor:
    return torch.zeros_like(query)


@torch.library.custom_op(
    "tt::sparse_matmul", mutates_args=[], device_types=["xla", "cpu"]
)
def sparse_matmul(
    input_tensor_a: torch.Tensor,
    input_tensor_b: torch.Tensor,
    sparsity: torch.Tensor,
    nnz: int = None,
    is_input_a_sparse: bool = False,
    is_input_b_sparse: bool = True,
) -> torch.Tensor:
    """
    Sparse matrix multiplication for MoE (Mixture of Experts) models.

    This operation performs matrix multiplication where computation is skipped
    for sparse (zero) blocks based on the sparsity tensor.

    Accepts flexible input formats for MoE data flow:
        - Gate-up (b_sparse): dispatch output [1, BD, S, H] or canonical [A, B, M, K]
        - Down (a_sparse): activation output [BD, S, E, inter] or canonical [A, E, M, K]
    Format conversion to canonical 4D is handled internally.

    Args:
        input_tensor_a: First input tensor (see above for accepted formats).
        input_tensor_b: Expert weights [1, E, K, N].
        sparsity: Sparsity mask tensor.
        nnz: Number of non-zero elements. If None, inferred at runtime.
        is_input_a_sparse: Whether input_tensor_a is sparse.
        is_input_b_sparse: Whether input_tensor_b is sparse.

    Returns:
        Output tensor. For auto-converted inputs, returns clean shapes:
            - Gate-up with dispatch input: [BD, S, E, N]
            - Down with activation input: [BD, S, E, N]
            - Otherwise: canonical sparse_matmul output shapes
    """
    device = input_tensor_a.device

    # Detect MoE inputs that need normalization to 4D canonical form.
    # Gate-up from dispatch: [1, BD, S, H] → [BD, S, 1, H]
    # Down from activation:  [BD, S, E, inter] → [BD*S, E, 1, inter]
    _moe_shape = None
    E_experts = input_tensor_b.shape[1]

    if not is_input_a_sparse and is_input_b_sparse:
        if input_tensor_a.dim() == 4 and input_tensor_a.shape[0] == 1:
            _, BD, S, H = input_tensor_a.shape
            _moe_shape = (BD, S)
    elif is_input_a_sparse and not is_input_b_sparse:
        # Detect MoE format [BD, S, E, inter] vs canonical [A, E, M, K].
        # Use sparsity dim: canonical has A == sparsity.shape[2] (reduced),
        # MoE format has BD != reduced. This works even when S == E_global.
        if input_tensor_a.dim() == 4 and input_tensor_a.shape[0] != sparsity.shape[2]:
            BD, S, _, _ = input_tensor_a.shape
            _moe_shape = (BD, S)

    if device.type == "xla":
        # Normalize MoE tensors to canonical sparse_matmul input formats only.
        # Hardware-driven tile decomposition is now handled by MLIR workarounds.
        if _moe_shape is not None:
            BD, S = _moe_shape
            if not is_input_a_sparse and is_input_b_sparse:
                # Gate-up: [1, BD, S, H] -> [BD, S, 1, H]
                input_tensor_a = input_tensor_a.permute(1, 2, 0, 3)
            elif is_input_a_sparse and not is_input_b_sparse:
                # Down: [BD, S, E, K] -> [BD*S, E, 1, K]
                E_in = input_tensor_a.shape[2]
                K_in = input_tensor_a.shape[-1]
                input_tensor_a = input_tensor_a.reshape(BD * S, E_in, 1, K_in)

        frontend_attributes = {
            "is_input_a_sparse": str(is_input_a_sparse),
            "is_input_b_sparse": str(is_input_b_sparse),
        }
        if nnz is not None:
            frontend_attributes["nnz"] = str(nnz)

        if is_input_a_sparse and is_input_b_sparse:
            output_shape = list(input_tensor_a.shape)
            output_shape[-1] = input_tensor_b.shape[-1]
        elif not is_input_a_sparse and is_input_b_sparse:
            A, B, M_dim, K = input_tensor_a.shape
            output_shape = [A, B, 1, E_experts, M_dim, input_tensor_b.shape[-1]]
        elif is_input_a_sparse and not is_input_b_sparse:
            output_shape = list(input_tensor_a.shape)
            output_shape[-1] = input_tensor_b.shape[-1]
        else:
            raise ValueError("Both sparse flags cannot be False")

        result = stablehlo_custom_call.stablehlo_custom_call(
            [input_tensor_a, input_tensor_b, sparsity],
            "tt.sparse_matmul",
            [output_shape],
            [input_tensor_a.dtype],
            frontend_attributes=frontend_attributes,
        )

        # Convert canonical sparse_matmul outputs back to clean MoE outputs.
        if _moe_shape is not None:
            BD, S = _moe_shape
            N = input_tensor_b.shape[-1]
            if not is_input_a_sparse and is_input_b_sparse:
                result = result.squeeze(2).squeeze(-2)
            elif is_input_a_sparse and not is_input_b_sparse:
                result = result.squeeze(2).reshape(BD, S, E_experts, N)

        return result

    elif device.type == "cpu":
        _t0 = time.perf_counter()
        _tiled = _moe_shape is not None
        if _tiled:
            BD, S = _moe_shape
            reduced = sparsity.shape[2]
            E_sp = sparsity.shape[-1]
            M = (BD * S) // reduced

            if not is_input_a_sparse and is_input_b_sparse:
                input_tensor_a = input_tensor_a.view(
                    BD, S // M, M, input_tensor_a.shape[-1]
                )
                sparsity = sparsity.view(BD, S // M, 1, E_sp)
            elif is_input_a_sparse and not is_input_b_sparse:
                E_in = input_tensor_a.shape[2]
                K_in = input_tensor_a.shape[-1]
                input_tensor_a = input_tensor_a.reshape(BD * S // M, M, E_in, K_in)
                input_tensor_a = input_tensor_a.permute(0, 2, 1, 3).contiguous()
                sparsity = sparsity.view(1, 1, BD * S // M, E_sp)

        orig_dtype = input_tensor_a.dtype
        input_tensor_a = input_tensor_a.float()
        sparsity = sparsity.float()
        input_b_casted = input_tensor_b.float()
        E = E_experts
        N = input_tensor_b.shape[-1]

        # Find active experts from sparsity to skip inactive ones
        if not (is_input_a_sparse and is_input_b_sparse):
            active_experts = set()
            sp_flat = sparsity.view(-1, E)
            for e in range(E):
                if sp_flat[:, e].any():
                    active_experts.add(e)

        if is_input_a_sparse and is_input_b_sparse:
            output = torch.matmul(input_tensor_a, input_b_casted)
            _perf_log("sparse_matmul_cpu_both", time.perf_counter() - _t0)
            return output.to(orig_dtype)

        elif not is_input_a_sparse and is_input_b_sparse:
            A, B, M_dim, K = input_tensor_a.shape
            output = torch.zeros(
                A, B, 1, E, M_dim, N, dtype=torch.float32, device=device
            )
            for e in active_experts:
                mask_e = sparsity[:, :, 0, e]
                out_e = torch.matmul(input_tensor_a, input_b_casted[0, e])
                output[:, :, 0, e, :, :] = out_e * mask_e.unsqueeze(-1).unsqueeze(-1)
            if _tiled:
                output = output.squeeze(2).permute(0, 1, 3, 2, 4).contiguous()
                output = output.view(BD, S, E, N)
            _perf_log("sparse_matmul_cpu_gate_up", time.perf_counter() - _t0)
            return output.to(orig_dtype)

        elif is_input_a_sparse and not is_input_b_sparse:
            A = input_tensor_a.shape[0]
            M_dim = input_tensor_a.shape[2]
            output = torch.zeros(A, E, M_dim, N, dtype=torch.float32, device=device)
            for e in active_experts:
                mask_e = sparsity[0, 0, :, e]
                out_e = torch.matmul(input_tensor_a[:, e], input_b_casted[0, e])
                output[:, e] = out_e * mask_e.unsqueeze(-1).unsqueeze(-1)
            if _tiled:
                output = output.view(BD, S // M, E, M, N)
                output = output.permute(0, 1, 3, 2, 4).contiguous()
                output = output.view(BD, S, E, N)
            _perf_log("sparse_matmul_cpu_down", time.perf_counter() - _t0)
            return output.to(orig_dtype)

        else:
            raise ValueError("Both sparse flags cannot be False")
    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@sparse_matmul.register_fake
def sparse_matmul_fake(
    input_tensor_a: torch.Tensor,
    input_tensor_b: torch.Tensor,
    sparsity: torch.Tensor,
    nnz: int = None,
    is_input_a_sparse: bool = False,
    is_input_b_sparse: bool = True,
) -> torch.Tensor:
    """FakeTensor implementation of sparse_matmul for torch dynamo tracing."""
    E = input_tensor_b.shape[1]
    N = input_tensor_b.shape[-1]

    # Detect MoE inputs (same logic as real op)
    _moe_shape = None
    if not is_input_a_sparse and is_input_b_sparse:
        if input_tensor_a.dim() == 4 and input_tensor_a.shape[0] == 1:
            _, BD, S, _ = input_tensor_a.shape
            _moe_shape = (BD, S)
    elif is_input_a_sparse and not is_input_b_sparse:
        if input_tensor_a.dim() == 4 and input_tensor_a.shape[1] != E:
            BD, S, _, _ = input_tensor_a.shape
            _moe_shape = (BD, S)

    if _moe_shape is not None:
        BD, S = _moe_shape
        output_shape = [BD, S, E, N]
    elif is_input_a_sparse and is_input_b_sparse:
        output_shape = list(input_tensor_a.shape)
        output_shape[-1] = N
    elif not is_input_a_sparse and is_input_b_sparse:
        A, B, M, K = input_tensor_a.shape
        output_shape = [A, B, 1, E, M, N]
    elif is_input_a_sparse and not is_input_b_sparse:
        output_shape = list(input_tensor_a.shape)
        output_shape[-1] = N
    else:
        raise ValueError("Both sparse flags cannot be False")

    return torch.zeros(
        output_shape, dtype=input_tensor_a.dtype, device=input_tensor_a.device
    )


@torch.library.custom_op(
    "tt::all_to_all_dispatch", mutates_args=[], device_types=["xla", "cpu"]
)
def all_to_all_dispatch(
    input_tensor: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int = 1,
    cluster_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch tokens to devices holding their selected experts.

    Selectively routes tokens based on expert_indices and expert_mapping,
    sending each token only to devices that hold its selected experts.

    Accepts flexible input formats:
        - input_tensor: [B, S, H] (3D) or [B, 1, S, H] (4D)
        - expert_indices: [B*S, K] (2D) or [B, S, K] (3D) or [B, 1, S, K] (4D)
        - expert_mapping: [1, 1, E, D]

    Returns:
        dispatched_tokens: [1, B*D, S, H] sparsely populated tokens
        expert_metadata: [1, B*D, S, K] all-gathered expert indices
    """
    device = input_tensor.device

    if device.type == "xla":
        # Keep frontend shape ops minimal for XLA path; rank normalization is
        # canonicalized in StableHLO->TTIR conversion.
        if input_tensor.dim() == 3:
            B, S, H = input_tensor.shape
        elif input_tensor.dim() == 4:
            B, _, S, H = input_tensor.shape
        else:
            raise ValueError(
                f"input_tensor must be rank 3 or 4, got {input_tensor.dim()}"
            )

        K = expert_indices.shape[-1]
        BD = B * num_devices
        output_shapes = [[1, BD, S, H], [1, BD, S, K]]
        output_dtypes = [input_tensor.dtype, expert_indices.dtype]

        frontend_attributes = {
            "num_devices": str(num_devices),
            "cluster_axis": str(cluster_axis),
        }

        return stablehlo_custom_call.stablehlo_custom_call(
            [input_tensor, expert_indices, expert_mapping],
            "tt.all_to_all_dispatch",
            output_shapes,
            output_dtypes,
            frontend_attributes=frontend_attributes,
        )

    elif device.type == "cpu":
        # Normalize to 4D [B, 1, S, H] for the CPU fallback kernel.
        if input_tensor.dim() == 3:
            B, S, H = input_tensor.shape
            input_tensor = input_tensor.unsqueeze(1)  # [B, 1, S, H]
        else:
            B, _, S, H = input_tensor.shape

        K = expert_indices.shape[-1]
        if expert_indices.dim() == 2:
            expert_indices = expert_indices.view(B, 1, S, K)
        elif expert_indices.dim() == 3:
            expert_indices = expert_indices.unsqueeze(1)  # [B, 1, S, K]

        _t0 = time.perf_counter()
        x = input_tensor.permute(1, 0, 2, 3)  # [1, B, S, H]
        m = expert_indices.permute(1, 0, 2, 3)  # [1, B, S, K]
        if num_devices > 1:
            x = x.repeat(1, num_devices, 1, 1)  # [1, B*D, S, H]
            m = m.repeat(1, num_devices, 1, 1)  # [1, B*D, S, K]
        _perf_log("dispatch_cpu", time.perf_counter() - _t0)
        return x.clone(), m.clone()

    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@all_to_all_dispatch.register_fake
def all_to_all_dispatch_fake(
    input_tensor: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int = 1,
    cluster_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_tensor.dim() == 3:
        B, S, H = input_tensor.shape
    else:
        B, _, S, H = input_tensor.shape
    K = expert_indices.shape[-1]
    BD = B * num_devices

    dispatched = torch.zeros(
        [1, BD, S, H], dtype=input_tensor.dtype, device=input_tensor.device
    )
    metadata = torch.zeros(
        [1, BD, S, K], dtype=expert_indices.dtype, device=expert_indices.device
    )
    return dispatched, metadata


@torch.library.custom_op(
    "tt::all_to_all_combine", mutates_args=[], device_types=["xla", "cpu"]
)
def all_to_all_combine(
    input_tensor: torch.Tensor,
    expert_metadata: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int = 1,
    cluster_axis: int = 0,
    num_experts_per_tok: int = 2,
    output_shard_dim: int = 1,
) -> torch.Tensor:
    """
    Combine expert outputs back to original token positions.

    Inverse of dispatch: gathers expert computation results from all devices
    and restores tokens to their original device and order.

    Accepts flexible input formats:
        - [E_local, B*D, S, H]: canonical (E first)
        - [B*D, S, E_local, H]: natural MoE output (E at dim -2), auto-permuted

    Args:
        input_tensor: Expert outputs (see above for accepted formats).
        expert_metadata: Routing metadata from dispatch [1, B*D, S, K], int64
        expert_mapping: One-hot expert-to-device mapping [1, 1, E, D], int64
        num_devices: Number of devices along dispatch axis (D)
        cluster_axis: Mesh axis to combine along (0=rows, 1=cols)
        num_experts_per_tok: Number of selected experts per token (K)
        output_shard_dim: Dimension index for the BD shard dimension (1 or 2).
            Auto-detected by the compiler; callers typically omit this.

    Returns:
        combined: [K, B, S, H]
    """
    device = input_tensor.device
    K = num_experts_per_tok

    if device.type == "xla":
        if input_tensor.dim() != 4:
            raise ValueError(f"input_tensor must be rank 4, got {input_tensor.dim()}")

        # Keep frontend shape ops minimal for XLA path; [BD, S, E, H] -> [E, BD, S, H]
        # normalization is canonicalized in StableHLO->TTIR conversion.
        E_global = expert_mapping.shape[2]  # [1, 1, E, D]
        D_total = expert_mapping.shape[3]
        E_candidates = {E_global} | ({E_global // D_total} if D_total > 1 else set())
        if input_tensor.shape[2] in E_candidates:
            BD, S, _, H = input_tensor.shape  # [BD, S, E, H]
        else:
            _, BD, S, H = input_tensor.shape  # [E, BD, S, H]

        B = BD // num_devices
        if output_shard_dim == 1:
            output_shape = [K, B, S, H]
        elif output_shard_dim == 2:
            output_shape = [K, S, B, H]
        else:
            raise ValueError(f"output_shard_dim must be 1 or 2, got {output_shard_dim}")

        frontend_attributes = {
            "num_devices": str(num_devices),
            "cluster_axis": str(cluster_axis),
            "num_experts_per_tok": str(K),
            "output_shard_dim": str(output_shard_dim),
        }

        return stablehlo_custom_call.stablehlo_custom_call(
            [input_tensor, expert_metadata, expert_mapping],
            "tt.all_to_all_combine",
            [output_shape],
            [input_tensor.dtype],
            frontend_attributes=frontend_attributes,
        )

    elif device.type == "cpu":
        # Keep existing CPU behavior: normalize [BD, S, E, H] to [E, BD, S, H].
        E_global = expert_mapping.shape[2]  # [1, 1, E, D]
        D = expert_mapping.shape[3]
        E_candidates = {E_global} | ({E_global // D} if D > 1 else set())
        if input_tensor.dim() == 4 and input_tensor.shape[2] in E_candidates:
            input_tensor = input_tensor.permute(2, 0, 1, 3)

        if output_shard_dim == 1:
            E_local, BD, S, H = input_tensor.shape
        elif output_shard_dim == 2:
            E_local, S, BD, H = input_tensor.shape
        else:
            raise ValueError(f"output_shard_dim must be 1 or 2, got {output_shard_dim}")

        _t0 = time.perf_counter()
        B_local = BD // num_devices
        metadata_indices = expert_metadata[0].long()  # [BD, S, K]

        if output_shard_dim == 1:
            # Vectorized: gather from input_tensor[expert_id, b, s, :] for each k
            # input_tensor: [E, BD, S, H], indices: [BD, S, K]
            combined = torch.zeros(
                K, B_local, S, H, dtype=input_tensor.dtype, device=device
            )
            for k in range(K):
                expert_ids = metadata_indices[:B_local, :, k]  # [B_local, S]
                # Clamp to valid range
                valid = (expert_ids >= 0) & (expert_ids < E_local)
                expert_ids_clamped = expert_ids.clamp(0, E_local - 1)
                # Advanced indexing: gather [B_local, S, H]
                b_idx = (
                    torch.arange(B_local, device=device)
                    .unsqueeze(1)
                    .expand_as(expert_ids)
                )
                s_idx = (
                    torch.arange(S, device=device).unsqueeze(0).expand_as(expert_ids)
                )
                gathered = input_tensor[
                    expert_ids_clamped, b_idx, s_idx, :
                ]  # [B_local, S, H]
                combined[k] = gathered * valid.unsqueeze(-1).to(gathered.dtype)
        else:
            combined = torch.zeros(
                K, S, B_local, H, dtype=input_tensor.dtype, device=device
            )
            for k in range(K):
                expert_ids = metadata_indices[:B_local, :, k]  # [B_local, S]
                valid = (expert_ids >= 0) & (expert_ids < E_local)
                expert_ids_clamped = expert_ids.clamp(0, E_local - 1)
                b_idx = (
                    torch.arange(B_local, device=device)
                    .unsqueeze(1)
                    .expand_as(expert_ids)
                )
                s_idx = (
                    torch.arange(S, device=device).unsqueeze(0).expand_as(expert_ids)
                )
                gathered = input_tensor[expert_ids_clamped, s_idx, b_idx, :]
                combined[k] = gathered.permute(1, 0, 2) * valid.permute(1, 0).unsqueeze(
                    -1
                ).to(gathered.dtype)

        _perf_log("combine_cpu", time.perf_counter() - _t0)
        return combined

    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@all_to_all_combine.register_fake
def all_to_all_combine_fake(
    input_tensor: torch.Tensor,
    expert_metadata: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int = 1,
    cluster_axis: int = 0,
    num_experts_per_tok: int = 2,
    output_shard_dim: int = 1,
) -> torch.Tensor:
    K = num_experts_per_tok

    if input_tensor.dim() != 4:
        raise ValueError(f"input_tensor must be rank 4, got {input_tensor.dim()}")

    # Mirror the XLA path shape inference without materializing frontend permute.
    E_global = expert_mapping.shape[2]  # [1, 1, E, D]
    D_total = expert_mapping.shape[3]
    E_candidates = {E_global} | ({E_global // D_total} if D_total > 1 else set())
    if input_tensor.shape[2] in E_candidates:
        BD, S, _, H = input_tensor.shape  # [BD, S, E, H]
    else:
        _, BD, S, H = input_tensor.shape  # [E, BD, S, H]

    B = BD // num_devices
    if output_shard_dim == 1:
        return torch.zeros(
            [K, B, S, H], dtype=input_tensor.dtype, device=input_tensor.device
        )
    if output_shard_dim == 2:
        return torch.zeros(
            [K, S, B, H], dtype=input_tensor.dtype, device=input_tensor.device
        )
    raise ValueError(f"output_shard_dim must be 1 or 2, got {output_shard_dim}")


@torch.library.custom_op(
    "tt::moe_expert_token_remap", mutates_args=[], device_types=["xla", "cpu"]
)
def moe_expert_token_remap(
    topk_tensor: torch.Tensor,
    expert_mapping: torch.Tensor,
    expert_metadata: torch.Tensor,
    num_devices: int = 1,
    reduction_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert global expert routing to local device expert mapping and sparsity.

    Accepts flexible topk_tensor formats:
        - [B*S, E] (2D): router scores, internally repeated for num_devices
        - [1, BD, S, E] (4D): pre-repeated (legacy)

    Returns:
        mapping: [1, BD, S, E], bfloat16
        reduced: [1, 1, ceil(BD*S/reduction_size), E], bfloat16
    """
    import math

    device = topk_tensor.device

    if device.type == "xla":
        # Keep frontend shape ops minimal for XLA path; rank normalization to
        # [1, BD, S, E] is canonicalized in StableHLO->TTIR conversion.
        BD = expert_metadata.shape[1]
        S = expert_metadata.shape[2]
        E = topk_tensor.shape[-1]
        reduced_seq = math.ceil(BD * S / reduction_size)

        output_shapes = [
            [1, BD, S, E],
            [1, 1, reduced_seq, E],
        ]
        output_dtypes = [topk_tensor.dtype, topk_tensor.dtype]

        frontend_attributes = {
            "reduction_size": str(reduction_size),
            "num_devices": str(num_devices),
        }

        return stablehlo_custom_call.stablehlo_custom_call(
            [topk_tensor, expert_mapping, expert_metadata],
            "tt.moe_expert_token_remap",
            output_shapes,
            output_dtypes,
            frontend_attributes=frontend_attributes,
        )

    # CPU fallback (vectorized)
    # Normalize to [1, BD, S, E] for the fallback implementation.
    if topk_tensor.dim() == 2:
        BD = expert_metadata.shape[1]
        S = expert_metadata.shape[2]
        B = BD // num_devices
        E = topk_tensor.shape[-1]
        topk_tensor = topk_tensor.view(B, S, E).repeat(num_devices, 1, 1).unsqueeze(0)
    elif topk_tensor.dim() == 3:
        topk_tensor = topk_tensor.repeat(num_devices, 1, 1).unsqueeze(0)

    D, BD, S, E = topk_tensor.shape
    K = expert_metadata.shape[-1]
    reduced_seq = math.ceil(BD * S / reduction_size)

    _t0 = time.perf_counter()
    mapping = torch.zeros(1, BD, S, E, dtype=topk_tensor.dtype, device=device)
    reduced = torch.zeros(1, 1, reduced_seq, E, dtype=topk_tensor.dtype, device=device)

    # expert_metadata: [D, BD, S, K] — selected expert indices
    # Scatter topk scores into mapping at the selected expert positions
    indices = expert_metadata.long()  # [D, BD, S, K]
    for d in range(D):
        # Gather scores for selected experts: [BD, S, K]
        scores = torch.gather(topk_tensor[d], dim=-1, index=indices[d])
        # Scatter into mapping: [1, BD, S, E]
        mapping[0].scatter_(-1, indices[d], scores)

    # Build reduced sparsity: any selected expert in each M-token chunk → 1.0
    token_idx = torch.arange(BD * S, device=device).view(BD, S)
    chunk_idx = token_idx // reduction_size  # [BD, S]
    # For each (chunk, expert) pair, mark as active
    for k_idx in range(K):
        expert_ids = indices[0, :, :, k_idx]  # [BD, S]
        flat_chunk = chunk_idx.reshape(-1)
        flat_expert = expert_ids.reshape(-1).long()
        reduced[0, 0, flat_chunk, flat_expert] = 1.0

    _perf_log("remap_cpu", time.perf_counter() - _t0)
    return mapping, reduced


@moe_expert_token_remap.register_fake
def moe_expert_token_remap_fake(
    topk_tensor: torch.Tensor,
    expert_mapping: torch.Tensor,
    expert_metadata: torch.Tensor,
    num_devices: int = 1,
    reduction_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    import math

    if topk_tensor.dim() == 2:
        BD = expert_metadata.shape[1]
        S = expert_metadata.shape[2]
        E = topk_tensor.shape[-1]
    elif topk_tensor.dim() == 3:
        BD = expert_metadata.shape[1]
        S = expert_metadata.shape[2]
        E = topk_tensor.shape[-1]
    else:
        _, BD, S, E = topk_tensor.shape

    reduced_seq = math.ceil(BD * S / reduction_size)

    mapping = torch.zeros(
        [1, BD, S, E], dtype=topk_tensor.dtype, device=topk_tensor.device
    )
    reduced = torch.zeros(
        [1, 1, reduced_seq, E],
        dtype=topk_tensor.dtype,
        device=topk_tensor.device,
    )
    return mapping, reduced


# Allow the torch dynamo to trace our custom operation(s). This will allow
# the tt custom operation(s) to be represented in a torch.fx.GraphModule.
for attr in dir(torch.ops.tt):
    # Filter out torch.ops.tt module attributes which are not ops.
    op = getattr(torch.ops.tt, attr)
    if isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
        torch.compiler.allow_in_graph(op)
