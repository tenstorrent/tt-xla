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

    # Handle shape requirements (same workaround as mark_argument_attributes)
    original_shape = list(tensor.shape)
    if len(tensor.shape) < 3:
        extra_dims = [1] * (3 - len(original_shape))
        tensor = tensor.reshape((*extra_dims, *original_shape))

    result = stablehlo_custom_call.stablehlo_custom_call(
        [tensor],
        "tt.sharding_constraint",  # tt-mlir converts this to sdy.sharding_constraint
        [tensor.shape],
        [tensor.dtype],
        frontend_attributes=frontend_attributes,
    )

    if len(original_shape) < 3:
        result = result.reshape(original_shape)

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

    Args:
        input_tensor_a: First input tensor. Shape depends on sparse mode:
            - is_input_a_sparse=True, is_input_b_sparse=True: [1, E, M, K]
            - is_input_a_sparse=False, is_input_b_sparse=True: [A, B, M, K]
            - is_input_a_sparse=True, is_input_b_sparse=False: [A, E, M, K]
        input_tensor_b: Second input tensor (expert weights). Shape:
            - [1, E, K, N] for all modes
        sparsity: Sparsity mask tensor (bfloat16, ROW_MAJOR). Shape depends on mode:
            - is_input_a_sparse=True, is_input_b_sparse=True: [1, 1, 1, E]
            - is_input_a_sparse=False, is_input_b_sparse=True: [A, B, 1, E]
            - is_input_a_sparse=True, is_input_b_sparse=False: [1, 1, A, E]
        nnz: Number of non-zero elements in sparsity tensor. If None, inferred at runtime.
        is_input_a_sparse: Whether input_tensor_a is sparse.
        is_input_b_sparse: Whether input_tensor_b is sparse.

    Returns:
        Output tensor with sparse results. Shape depends on mode:
            - is_input_a_sparse=True, is_input_b_sparse=True: [1, E, M, N]
            - is_input_a_sparse=False, is_input_b_sparse=True: [A, B, 1, E, M, N]
            - is_input_a_sparse=True, is_input_b_sparse=False: [A, E, M, N]
    """
    device = input_tensor_a.device

    if device.type == "xla":
        frontend_attributes = {
            "is_input_a_sparse": str(is_input_a_sparse),
            "is_input_b_sparse": str(is_input_b_sparse),
        }
        if nnz is not None:
            frontend_attributes["nnz"] = str(nnz)

        # Compute output shape based on sparse mode
        if is_input_a_sparse and is_input_b_sparse:
            # [1, E, M, K] @ [1, E, K, N] -> [1, E, M, N]
            output_shape = list(input_tensor_a.shape)
            output_shape[-1] = input_tensor_b.shape[-1]
        elif not is_input_a_sparse and is_input_b_sparse:
            # [A, B, M, K] @ [1, E, K, N] -> [A, B, 1, E, M, N]
            A, B, M, K = input_tensor_a.shape
            E = input_tensor_b.shape[1]
            N = input_tensor_b.shape[-1]
            output_shape = [A, B, 1, E, M, N]
        elif is_input_a_sparse and not is_input_b_sparse:
            # [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
            output_shape = list(input_tensor_a.shape)
            output_shape[-1] = input_tensor_b.shape[-1]
        else:
            raise ValueError(
                "Invalid sparse mode: both is_input_a_sparse and is_input_b_sparse cannot be False"
            )

        return stablehlo_custom_call.stablehlo_custom_call(
            [input_tensor_a, input_tensor_b, sparsity],
            "tt.sparse_matmul",
            [output_shape],
            [input_tensor_a.dtype],
            frontend_attributes=frontend_attributes,
        )

    elif device.type == "cpu":
        # CPU fallback: loop over experts to avoid broadcasting weights
        # across large batch dimensions (can exceed 1TB for D=8, E=32).
        input_b_casted = input_tensor_b.to(input_tensor_a.dtype)

        if is_input_a_sparse and is_input_b_sparse:
            # [1, E, M, K] @ [1, E, K, N] -> [1, E, M, N]
            E = input_tensor_b.shape[1]
            N = input_tensor_b.shape[-1]
            M = input_tensor_a.shape[2]
            output = torch.zeros(1, E, M, N, dtype=input_tensor_a.dtype, device=device)
            for e in range(E):
                if sparsity[0, 0, 0, e] > 0:
                    output[0, e] = torch.matmul(
                        input_tensor_a[0, e], input_b_casted[0, e]
                    )
            return output

        elif not is_input_a_sparse and is_input_b_sparse:
            # [A, B, M, K] @ [1, E, K, N] -> [A, B, 1, E, M, N]
            A, B, M, K = input_tensor_a.shape
            E = input_tensor_b.shape[1]
            N = input_tensor_b.shape[-1]
            output = torch.zeros(
                A, B, 1, E, M, N, dtype=input_tensor_a.dtype, device=device
            )
            for e in range(E):
                mask_e = sparsity[:, :, 0, e]  # [A, B]
                if mask_e.any():
                    # [A, B, M, K] @ [K, N] -> [A, B, M, N]
                    out_e = torch.matmul(input_tensor_a, input_b_casted[0, e])
                    output[:, :, 0, e, :, :] = out_e * mask_e.unsqueeze(-1).unsqueeze(
                        -1
                    )
            return output

        elif is_input_a_sparse and not is_input_b_sparse:
            # [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
            A = input_tensor_a.shape[0]
            E = input_tensor_b.shape[1]
            M = input_tensor_a.shape[2]
            N = input_tensor_b.shape[-1]
            output = torch.zeros(A, E, M, N, dtype=input_tensor_a.dtype, device=device)
            for e in range(E):
                mask_e = sparsity[0, 0, :, e]  # [A]
                if mask_e.any():
                    # [A, M, K] @ [K, N] -> [A, M, N]
                    out_e = torch.matmul(input_tensor_a[:, e], input_b_casted[0, e])
                    output[:, e] = out_e * mask_e.unsqueeze(-1).unsqueeze(-1)
            return output

        else:
            raise ValueError(
                "Invalid sparse mode: both is_input_a_sparse and is_input_b_sparse cannot be False"
            )
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
    if is_input_a_sparse and is_input_b_sparse:
        output_shape = list(input_tensor_a.shape)
        output_shape[-1] = input_tensor_b.shape[-1]
    elif not is_input_a_sparse and is_input_b_sparse:
        A, B, M, K = input_tensor_a.shape
        E = input_tensor_b.shape[1]
        N = input_tensor_b.shape[-1]
        output_shape = [A, B, 1, E, M, N]
    elif is_input_a_sparse and not is_input_b_sparse:
        output_shape = list(input_tensor_a.shape)
        output_shape[-1] = input_tensor_b.shape[-1]
    else:
        raise ValueError(
            "Invalid sparse mode: both is_input_a_sparse and is_input_b_sparse cannot be False"
        )

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

    Args:
        input_tensor: Input tokens [B, 1, S, H], bfloat16
        expert_indices: Selected expert IDs per token [B, 1, S, K], int64
        expert_mapping: One-hot expert-to-device mapping [1, 1, E, D], int64
        num_devices: Number of devices along dispatch axis (D)
        cluster_axis: Mesh axis to dispatch along (0=rows, 1=cols)

    Returns:
        dispatched_tokens: [1, B*D, S, H] sparsely populated tokens
        expert_metadata: [1, B*D, S, K] all-gathered expert indices
    """
    device = input_tensor.device
    B, _, S, H = input_tensor.shape
    K = expert_indices.shape[-1]

    if device.type == "xla":
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
        # CPU fallback: simulate dispatch by repeating tokens D times.
        # Shape must match fake kernel: [1, B*D, S, H] and [1, B*D, S, K].
        # On real hardware, dispatch selectively routes tokens; on CPU we
        # replicate so that downstream sparse_matmul sees all tokens.
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
) -> torch.Tensor:
    """
    Combine expert outputs back to original token positions.

    Inverse of dispatch: gathers expert computation results from all devices
    and restores tokens to their original device and order.

    Args:
        input_tensor: Expert outputs [E_local, B*D, S, H], bfloat16
        expert_metadata: Routing metadata from dispatch [1, B*D, S, K], int64
        expert_mapping: One-hot expert-to-device mapping [1, 1, E, D], int64
        num_devices: Number of devices along dispatch axis (D)
        cluster_axis: Mesh axis to combine along (0=rows, 1=cols)
        num_experts_per_tok: Number of selected experts per token (K)

    Returns:
        combined: [K, B, S, H] expert outputs restored to original positions
    """
    device = input_tensor.device
    E_local, BD, S, H = input_tensor.shape
    K = num_experts_per_tok
    B = BD // num_devices

    if device.type == "xla":
        output_shape = [K, B, S, H]

        frontend_attributes = {
            "num_devices": str(num_devices),
            "cluster_axis": str(cluster_axis),
            "num_experts_per_tok": str(K),
        }

        return stablehlo_custom_call.stablehlo_custom_call(
            [input_tensor, expert_metadata, expert_mapping],
            "tt.all_to_all_combine",
            [output_shape],
            [input_tensor.dtype],
            frontend_attributes=frontend_attributes,
        )

    elif device.type == "cpu":
        # CPU fallback: dispatch repeats tokens D times, so BD = B * D.
        # Combine reverses this by taking only the first B entries (all
        # D copies are identical on CPU since dispatch just replicates).
        B_local = BD // num_devices
        metadata_indices = expert_metadata[0]  # [BD, S, K]
        combined = torch.zeros(
            K, B_local, S, H, dtype=input_tensor.dtype, device=device
        )

        for b in range(B_local):
            for s in range(S):
                for k in range(K):
                    expert_id = metadata_indices[b, s, k].item()
                    if 0 <= expert_id < E_local:
                        combined[k, b, s, :] = input_tensor[expert_id, b, s, :]

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
) -> torch.Tensor:
    _, BD, S, H = input_tensor.shape
    K = num_experts_per_tok
    B = BD // num_devices

    return torch.zeros(
        [K, B, S, H], dtype=input_tensor.dtype, device=input_tensor.device
    )


# Allow the torch dynamo to trace our custom operation(s). This will allow
# the tt custom operation(s) to be represented in a torch.fx.GraphModule.
for attr in dir(torch.ops.tt):
    # Filter out torch.ops.tt module attributes which are not ops.
    op = getattr(torch.ops.tt, attr)
    if isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
        torch.compiler.allow_in_graph(op)
