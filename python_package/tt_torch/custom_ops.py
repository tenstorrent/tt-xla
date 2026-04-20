# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union

import torch
from torch_xla.experimental import stablehlo_custom_call


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
    "tt::weight_dtype_override", mutates_args=[], device_types=["cpu", "xla"]
)
def weight_dtype_override(tensor: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """
    Apply a per-tensor weight dtype constraint for tt-mlir's weight dtype conversion pass.

    This function is a custom registered operator accessible as torch.ops.tt.weight_dtype_override.
    It creates a stablehlo.custom_call @tt.weight_dtype_override op whose frontend_attributes
    carry the target dtype. The C++ frontend pass lifts this to a function arg attribute, and
    the tt-mlir weight dtype conversion pass reads it to insert a typecast for this specific weight.

    Args:
        tensor: The weight tensor to annotate with a target dtype
        dtype_str: Target dtype string, one of "bfp_bf4", "bfp_bf8", or "bf16"

    Returns:
        The tensor with weight dtype constraint metadata applied
    """
    if tensor.device.type == "cpu":
        return tensor.clone()

    assert isinstance(
        dtype_str, str
    ), f"dtype_str must be a string, received {type(dtype_str)}"
    assert dtype_str in [
        "bfp_bf4",
        "bfp_bf8",
        "bf16",
    ], f"dtype_str must be one of 'bfp_bf4', 'bfp_bf8', or 'bf16', received {dtype_str}"

    frontend_attributes = {"ttcore.weight_dtype": dtype_str}

    # stablehlo_custom_call causes issues within XLA for shapes which are 2D or less.
    # Workaround: reshape the tensor to 3D, then reshape back after the custom call.
    original_shape = list(tensor.shape)
    if len(tensor.shape) < 3:
        extra_dims = [1] * (3 - len(original_shape))
        tensor = tensor.reshape((*extra_dims, *original_shape))

    result = stablehlo_custom_call.stablehlo_custom_call(
        [tensor],
        "tt.weight_dtype_override",
        [tensor.shape],
        [tensor.dtype],
        frontend_attributes=frontend_attributes,
    )

    if len(original_shape) < 3:
        result = result.reshape(original_shape)

    return result


@weight_dtype_override.register_fake
def _(tensor: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """
    FakeTensor implementation of torch.ops.tt.weight_dtype_override.
    This must be implemented in order for dynamo to trace the function.
    """
    return tensor.clone()


@weight_dtype_override.register_autograd
def _(ctx, grad_output):
    """
    Autograd implementation for weight_dtype_override.
    This op only applies dtype metadata, so gradients pass through unchanged.
    Returns gradients for: (tensor, dtype_str)
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
    sliding_window_size: int = None,
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

        if sliding_window_size is not None:
            assert (
                sliding_window_size > 0
            ), f"sliding_window_size must be a positive integer, but is {sliding_window_size}"
            frontend_attributes["sliding_window_size"] = str(sliding_window_size)

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
    sliding_window_size: int = None,
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
        if input_tensor_a.dim() == 4 and input_tensor_a.shape[1] != E_experts:
            BD, S, _, _ = input_tensor_a.shape
            _moe_shape = (BD, S)

    if device.type == "xla":
        # Pre-tile MoE tensors so the M (tile) dimension is already 32-aligned.
        # This avoids MLIR workaround reshape/permute overhead entirely.
        #
        # Gate-up: [1, BD, S, H] → [BD, S//M, M, H]  (reshape only)
        #   output: [BD, S//M, 1, E, M, N] → squeeze → [BD, S//M, E, M, N] (5D)
        #   caller does bias add on 5D, activation, then reshapes for down.
        #
        # Down: caller sends [BD*S//M, E, M, inter] (already canonical [A,E,M,K])
        #   → not detected as _moe_shape, passes through directly
        #   output: [BD*S//M, E, M, H] — caller does bias add, permute, reshape.
        if _moe_shape is not None:
            BD, S = _moe_shape
            reduced = sparsity.shape[2]
            M = (BD * S) // reduced

            if not is_input_a_sparse and is_input_b_sparse:
                H = input_tensor_a.shape[-1]
                split_seq = S % M == 0 and S >= M
                if split_seq:
                    input_tensor_a = input_tensor_a.reshape(BD, S // M, M, H)
                    sparsity = sparsity.reshape(BD, S // M, 1, E_experts)
                else:
                    # Decode: tile on BD instead
                    input_tensor_a = input_tensor_a.reshape(BD // M, M, S, H)
                    input_tensor_a = input_tensor_a.permute(0, 2, 1, 3)
                    sparsity = sparsity.reshape(BD // M, S, 1, E_experts)

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

        # Convert tiled sparse_matmul outputs.
        if _moe_shape is not None:
            if not is_input_a_sparse and is_input_b_sparse:
                # [A, B, 1, E, M, N] → [A, B, E, M, N] (5D tiled)
                result = result.squeeze(2).squeeze(-2)

        return result

    elif device.type == "cpu":
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
            # A2aSparseMLP dispatch layout: [1, BD, S, H]
            _, BD, S, _ = input_tensor_a.shape
            _moe_shape = (BD, S)
        elif input_tensor_a.dim() == 4 and input_tensor_a.shape[2] == 1:
            # SparseMLP layout: [BD, S, 1, H]
            BD, S, _, _ = input_tensor_a.shape
            _moe_shape = (BD, S)
    elif is_input_a_sparse and not is_input_b_sparse:
        if input_tensor_a.dim() == 4 and input_tensor_a.shape[1] != E:
            BD, S, _, _ = input_tensor_a.shape
            _moe_shape = (BD, S)

    if _moe_shape is not None:
        BD, S = _moe_shape
        reduced = sparsity.shape[2]
        M = (BD * S) // reduced
        if not is_input_a_sparse and is_input_b_sparse:
            # Gate-up: tiled output [A, B, E, M, N] (5D)
            split_seq = S % M == 0 and S >= M
            A = BD if split_seq else BD // M
            B = S // M if split_seq else S
            output_shape = [A, B, E, M, N]
        else:
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
        # Canonicalize inputs to 4D for runtime compatibility.
        if input_tensor.dim() == 3:
            B, S, H = input_tensor.shape
            input_tensor = input_tensor.reshape(B, 1, S, H)
        elif input_tensor.dim() == 4:
            B, _, S, H = input_tensor.shape
        else:
            raise ValueError(
                f"input_tensor must be rank 3 or 4, got {input_tensor.dim()}"
            )

        # Canonicalize expert_indices to 4D [B, 1, S, K]
        K = expert_indices.shape[-1]
        if expert_indices.dim() == 2:
            expert_indices = expert_indices.reshape(B, S, K).unsqueeze(1)
        elif expert_indices.dim() == 3:
            expert_indices = expert_indices.unsqueeze(1)

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

        x = input_tensor.permute(1, 0, 2, 3)  # [1, B, S, H]
        m = expert_indices.permute(1, 0, 2, 3)  # [1, B, S, K]
        if num_devices > 1:
            x = x.repeat(1, num_devices, 1, 1)  # [1, B*D, S, H]
            m = m.repeat(1, num_devices, 1, 1)  # [1, B*D, S, K]
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


def _a2a_dispatch_setup_context(ctx, inputs, output):
    input_tensor = inputs[0]
    expert_indices = inputs[1]
    expert_mapping = inputs[2]
    num_devices = inputs[3]
    cluster_axis = inputs[4]

    ctx.save_for_backward(expert_indices, expert_mapping)
    ctx.input_shape = tuple(input_tensor.shape)
    ctx.num_devices = int(num_devices)
    ctx.cluster_axis = int(cluster_axis)

    _, metadata = output
    ctx.mark_non_differentiable(metadata)


def _a2a_dispatch_backward(ctx, grad_dispatched, grad_metadata):
    """Adjoint of all_to_all_dispatch implemented via all_to_all_combine.

    On XLA (multi-chip), combine is the reverse all-to-all of dispatch. We
    call combine with the same routing (metadata rebuilt from user-supplied
    expert_indices, same expert_mapping/num_devices/cluster_axis) then sum
    over K to get the per-token gradient.

    For combine's kernel to accept the input we need E_local == E / D (same
    shape convention the MoE forward uses, see sparse_mlp.py). We materialize
    that by tiling grad_dispatched along a new leading E_local dim, in the
    [E_local, 1, BD*S, H] layout that the kernel and the SPMD sharding
    pattern expect.

    CPU / single-device: combine's CPU fallback is a local gather-with-clamp
    that doesn't realise the adjoint for our shape; we fall back to the math
    adjoint (sum over D copies).
    """
    expert_indices, expert_mapping = ctx.saved_tensors
    input_shape = ctx.input_shape
    D = ctx.num_devices
    cluster_axis = ctx.cluster_axis

    input_was_3d = len(input_shape) == 3
    if input_was_3d:
        B, S, H = input_shape
    else:
        B, _, S, H = input_shape

    # combine needs SPMD + proper mesh sharding; without them the TTNN
    # kernel crashes in fabric setup. Fall back to the math adjoint whenever
    # we're on CPU, num_devices==1, or SPMD isn't enabled on this run.
    _use_combine = False
    if D > 1 and grad_dispatched.device.type == "xla":
        try:
            import torch_xla.runtime as _xr  # local import — avoid at module top

            _use_combine = _xr.is_spmd()
        except Exception:
            _use_combine = False

    if not _use_combine:
        g = grad_dispatched.reshape(1, D, B, S, H).sum(dim=1)
        g = g.permute(1, 0, 2, 3).contiguous()
        if input_was_3d:
            g = g.reshape(B, S, H)
        return g, None, None, None, None

    K = expert_indices.shape[-1]
    E = expert_mapping.shape[2]
    assert E % D == 0, f"E ({E}) must be divisible by num_devices ({D}) for combine-based dispatch bwd"
    E_local = E // D

    # Build combine's metadata [1, 1, BD*S, K] from expert_indices (user
    # input) — NOT from the forward's tuple output, which would chain the
    # dispatch tuple into the backward graph and trip Shardy.
    ei = expert_indices
    if ei.dim() == 2:
        ei = ei.reshape(B, S, K).unsqueeze(1)
    elif ei.dim() == 3:
        ei = ei.unsqueeze(1)
    metadata_flat = ei.permute(1, 0, 2, 3)
    if D > 1:
        metadata_flat = metadata_flat.repeat(1, D, 1, 1)
    metadata_flat = metadata_flat.reshape(1, 1, B * D * S, K).contiguous()

    # Reshape grad_dispatched [1, BD, S, H] → [1, 1, BD*S, H] and tile the
    # leading dim to E_local so combine's kernel sees its expected layout.
    g_flat = grad_dispatched.reshape(1, 1, B * D * S, H)
    g_tiled = g_flat.expand(E_local, 1, B * D * S, H).contiguous()

    combined = torch.ops.tt.all_to_all_combine(
        g_tiled,  # [E_local, 1, BD*S, H]
        metadata_flat,  # [1, 1, BD*S, K]
        expert_mapping,  # [1, 1, E, D]
        num_devices=D,
        cluster_axis=cluster_axis,
        num_experts_per_tok=K,
        output_shard_dim=2,  # [K, 1, B*S, H]
    )
    # combined: [K, 1, B*S, H]. Sum K → per-token gradient.
    d_input = combined.sum(dim=0).reshape(B, S, H)
    if not input_was_3d:
        d_input = d_input.reshape(B, 1, S, H)
    return d_input, None, None, None, None


all_to_all_dispatch.register_autograd(
    _a2a_dispatch_backward,
    setup_context=_a2a_dispatch_setup_context,
)


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

        # metadata is [1, 1, tokens, K] where tokens = BD*S.
        tokens = expert_metadata.shape[2]
        H = input_tensor.shape[-1]
        tokens_per_device = tokens // num_devices

        if output_shard_dim == 1:
            output_shape = [K, tokens_per_device, 1, H]
        elif output_shard_dim == 2:
            output_shape = [K, 1, tokens_per_device, H]
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
        # metadata is [1, 1, tokens, K] where tokens = BD*S.
        tokens = expert_metadata.shape[2]
        H = input_tensor.shape[-1]
        tokens_per_device = tokens // num_devices

        # Normalize input to [E, tokens, H] for indexing.
        E_local = input_tensor.shape[0]
        input_flat = input_tensor.reshape(E_local, tokens, H)

        # metadata indices: [1, 1, tokens, K] → [tokens, K]
        metadata_indices = expert_metadata[0, 0].long()  # [tokens, K]

        combined = torch.zeros(
            K, tokens_per_device, H, dtype=input_tensor.dtype, device=device
        )
        for k in range(K):
            expert_ids = metadata_indices[:tokens_per_device, k]  # [tokens_per_device]
            valid = (expert_ids >= 0) & (expert_ids < E_local)
            expert_ids_clamped = expert_ids.clamp(0, E_local - 1)
            t_idx = torch.arange(tokens_per_device, device=device)
            gathered = input_flat[
                expert_ids_clamped, t_idx, :
            ]  # [tokens_per_device, H]
            combined[k] = gathered * valid.unsqueeze(-1).to(gathered.dtype)

        # Reshape to match output_shard_dim format: [K, tokens_per_device, H] → [K, 1, tpd, H]
        if output_shard_dim == 2:
            combined = combined.unsqueeze(1)  # [K, 1, tokens_per_device, H]
        else:
            combined = combined.unsqueeze(2)  # [K, tokens_per_device, 1, H]

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

    # metadata is [1, 1, tokens, K].
    tokens = expert_metadata.shape[2]
    H = input_tensor.shape[-1]
    tokens_per_device = tokens // num_devices

    if output_shard_dim == 1:
        return torch.zeros(
            [K, tokens_per_device, 1, H],
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
    if output_shard_dim == 2:
        return torch.zeros(
            [K, 1, tokens_per_device, H],
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
    raise ValueError(f"output_shard_dim must be 1 or 2, got {output_shard_dim}")


def _a2a_combine_setup_context(ctx, inputs, output):
    # inputs: (input_tensor, expert_metadata, expert_mapping, num_devices,
    #          cluster_axis, num_experts_per_tok, output_shard_dim)
    input_tensor = inputs[0]
    expert_metadata = inputs[1]
    num_devices = inputs[3]
    K = inputs[5]
    output_shard_dim = inputs[6]

    ctx.save_for_backward(expert_metadata)
    ctx.input_shape = tuple(input_tensor.shape)
    ctx.num_devices = int(num_devices)
    ctx.K = int(K)
    ctx.output_shard_dim = int(output_shard_dim)


def _a2a_combine_backward(ctx, grad_combined):
    """Adjoint of all_to_all_combine.

    Forward: combined[k, tok, :] = input_flat[metadata[tok, k], tok, :] * valid(tok, k).
    Adjoint: d_input_flat[e, tok, :] = sum_{k: metadata[tok, k]==e and valid} d_combined[k, tok, :].

    Implemented with torch's native scatter_add — no new custom op is
    introduced. On multi-chip hardware the compiler is responsible for
    inserting the appropriate cross-device collective if the expert dim is
    sharded; under replicated execution the scatter is self-contained per
    chip and produces the same result everywhere.
    """
    (expert_metadata,) = ctx.saved_tensors
    input_shape = ctx.input_shape
    num_devices = ctx.num_devices
    K = ctx.K
    output_shard_dim = ctx.output_shard_dim

    E_local = input_shape[0]
    tokens = expert_metadata.shape[2]
    H = input_shape[-1]
    tokens_per_device = tokens // num_devices

    # Undo shard_dim insertion from forward to recover [K, tpd, H].
    if output_shard_dim == 1:
        g = grad_combined.squeeze(2)
    elif output_shard_dim == 2:
        g = grad_combined.squeeze(1)
    else:
        raise ValueError(f"output_shard_dim must be 1 or 2, got {output_shard_dim}")

    metadata_indices = expert_metadata[0, 0, :tokens_per_device].long()  # [tpd, K]
    expert_ids_KT = metadata_indices.transpose(0, 1).contiguous()  # [K, tpd]
    valid_KT = (expert_ids_KT >= 0) & (expert_ids_KT < E_local)
    expert_ids_clamped = expert_ids_KT.clamp(0, E_local - 1)

    g_masked = g * valid_KT.unsqueeze(-1).to(g.dtype)

    idx = expert_ids_clamped.unsqueeze(-1).expand(K, tokens_per_device, H)
    partial = torch.zeros(
        E_local, tokens_per_device, H, dtype=g.dtype, device=g.device
    )
    partial.scatter_add_(0, idx, g_masked)

    if tokens > tokens_per_device:
        zero_pad = torch.zeros(
            E_local,
            tokens - tokens_per_device,
            H,
            dtype=g.dtype,
            device=g.device,
        )
        d_input_flat = torch.cat([partial, zero_pad], dim=1)
    else:
        d_input_flat = partial

    d_input = d_input_flat.reshape(input_shape)

    return d_input, None, None, None, None, None, None


all_to_all_combine.register_autograd(
    _a2a_combine_backward,
    setup_context=_a2a_combine_setup_context,
)


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
        # metadata is [1, 1, tokens, K] where tokens = BD*S.
        tokens = expert_metadata.shape[2]
        E = topk_tensor.shape[-1]
        reduced_seq = math.ceil(tokens / reduction_size)

        # Canonicalize topk_tensor to 4D [1, 1, tokens, E].
        if topk_tensor.dim() == 2:
            BS = topk_tensor.shape[0]
            topk_tensor = topk_tensor.unsqueeze(0)  # [1, B*S, E]
            if num_devices > 1:
                topk_tensor = topk_tensor.repeat(1, num_devices, 1)  # [1, BD*S, E]
            topk_tensor = topk_tensor.unsqueeze(0)  # [1, 1, tokens, E]
        elif topk_tensor.dim() == 3:
            if num_devices > 1:
                topk_tensor = topk_tensor.repeat(1, num_devices, 1)
            topk_tensor = topk_tensor.unsqueeze(0)

        output_shapes = [
            [1, 1, tokens, E],
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
    # metadata is [1, 1, tokens, K] where tokens = BD*S.
    tokens = expert_metadata.shape[2]
    K = expert_metadata.shape[-1]
    E = topk_tensor.shape[-1]
    reduced_seq = math.ceil(tokens / reduction_size)

    # Normalize topk_tensor to [1, 1, tokens, E].
    if topk_tensor.dim() == 2:
        BS = topk_tensor.shape[0]
        topk_tensor = topk_tensor.unsqueeze(0)  # [1, B*S, E]
        if num_devices > 1:
            topk_tensor = topk_tensor.repeat(1, num_devices, 1)
        topk_tensor = topk_tensor.unsqueeze(0)  # [1, 1, tokens, E]
    elif topk_tensor.dim() == 3:
        if num_devices > 1:
            topk_tensor = topk_tensor.repeat(1, num_devices, 1)
        topk_tensor = topk_tensor.unsqueeze(0)

    D, _, tokens_dim, E = topk_tensor.shape

    mapping = torch.zeros(1, 1, tokens, E, dtype=topk_tensor.dtype, device=device)
    reduced = torch.zeros(1, 1, reduced_seq, E, dtype=topk_tensor.dtype, device=device)

    # expert_metadata: [1, 1, tokens, K] — selected expert indices
    # topk_tensor: [1, 1, tokens, E] — router scores
    indices = expert_metadata.long()  # [1, 1, tokens, K]
    # Gather scores for selected experts and scatter into mapping
    scores = torch.gather(topk_tensor[0, 0], dim=-1, index=indices[0, 0])
    mapping[0, 0].scatter_(-1, indices[0, 0], scores)

    # Build reduced sparsity: any selected expert in each M-token chunk → 1.0
    chunk_idx = torch.arange(tokens, device=device) // reduction_size
    for k_idx in range(K):
        expert_ids = indices[0, 0, :, k_idx]  # [tokens]
        reduced[0, 0, chunk_idx, expert_ids.long()] = 1.0

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

    # metadata is [1, 1, tokens, K]
    tokens = expert_metadata.shape[2]
    E = topk_tensor.shape[-1]
    reduced_seq = math.ceil(tokens / reduction_size)

    mapping = torch.zeros(
        [1, 1, tokens, E], dtype=topk_tensor.dtype, device=topk_tensor.device
    )
    reduced = torch.zeros(
        [1, 1, reduced_seq, E],
        dtype=topk_tensor.dtype,
        device=topk_tensor.device,
    )
    return mapping, reduced


def _moe_remap_setup_context(ctx, inputs, output):
    # inputs: (topk_tensor, expert_mapping, expert_metadata, num_devices, reduction_size)
    topk_tensor = inputs[0]
    expert_metadata = inputs[2]
    num_devices = inputs[3]

    ctx.save_for_backward(expert_metadata)
    ctx.topk_shape = tuple(topk_tensor.shape)
    ctx.num_devices = int(num_devices)

    # `reduced` is a 0/1 sparsity indicator built from integer ops; no grad flows through it.
    _mapping, reduced = output
    ctx.mark_non_differentiable(reduced)


def _moe_remap_backward(ctx, grad_mapping, grad_reduced):
    """Adjoint of moe_expert_token_remap.

    Forward (after canonicalizing topk to [1, 1, tokens, E]):
        mapping[0, 0, tok, e] = topk[tok, e]  if e ∈ {metadata[tok, k] : k in [K)}
                                      else 0
        reduced[...] = 0/1 sparsity (integer-only; no float gradient)

    Adjoint w.r.t. the canonicalized topk:
        d_topk[tok, e] = d_mapping[0, 0, tok, e]  if e is selected for tok
                                 else 0

    We realize this as: gather d_mapping values at the K selected positions
    per token, scatter them into a zero tensor of shape [tokens, E]. Then
    undo the canonicalization (squeeze/unsqueeze + reduce over the num_devices
    replication) to restore the original topk shape.
    """
    (expert_metadata,) = ctx.saved_tensors
    topk_shape = ctx.topk_shape
    num_devices = ctx.num_devices

    # Strip leading [1, 1] from grad_mapping → [tokens, E]
    g = grad_mapping[0, 0]

    indices = expert_metadata[0, 0].long()  # [tokens, K]
    scores = torch.gather(g, dim=-1, index=indices)  # [tokens, K]
    d_topk_flat = torch.zeros_like(g)
    d_topk_flat.scatter_(-1, indices, scores)  # [tokens, E]

    # Undo the forward's shape canonicalization.
    E = topk_shape[-1]
    if len(topk_shape) == 2:
        # Forward: [BS, E] → unsqueeze → repeat(D) → unsqueeze → [1, 1, tokens, E]
        BS = topk_shape[0]
        if num_devices > 1:
            d_topk = d_topk_flat.reshape(num_devices, BS, E).sum(dim=0)
        else:
            d_topk = d_topk_flat
    elif len(topk_shape) == 3:
        # Forward: [A, B, E] → repeat(D) on dim 1 → unsqueeze → [1, A, D*B, E]
        A = topk_shape[0]
        B = topk_shape[1]
        tokens = d_topk_flat.shape[0]
        # [tokens, E] is actually flattened [A, D*B, E] when A != 1, or just [D*B, E] when A==1.
        if A * B * num_devices != tokens:
            raise ValueError(
                f"topk_shape {topk_shape} inconsistent with tokens={tokens}, D={num_devices}"
            )
        reshaped = d_topk_flat.reshape(A, num_devices, B, E)
        d_topk = reshaped.sum(dim=1) if num_devices > 1 else reshaped.squeeze(1)
    elif len(topk_shape) == 4:
        # Forward: already [1, BD, S, E] (legacy pre-repeated); just reshape back.
        d_topk = d_topk_flat.reshape(topk_shape)
    else:
        raise ValueError(f"Unsupported topk rank: {len(topk_shape)}")

    # Return grads for: (topk_tensor, expert_mapping, expert_metadata, num_devices, reduction_size)
    return d_topk, None, None, None, None


moe_expert_token_remap.register_autograd(
    _moe_remap_backward,
    setup_context=_moe_remap_setup_context,
)


# Allow the torch dynamo to trace our custom operation(s). This will allow
# the tt custom operation(s) to be represented in a torch.fx.GraphModule.
for attr in dir(torch.ops.tt):
    # Filter out torch.ops.tt module attributes which are not ops.
    op = getattr(torch.ops.tt, attr)
    if isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
        torch.compiler.allow_in_graph(op)
