# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
    ), "query must be a 4D tensor: [1, B, num_heads, head_size]."
    assert (
        len(key.shape) == 4
    ), "key must be a 4D tensor: [B, num_kv_heads, seq_len, head_size]."
    assert (
        len(value.shape) == 4
    ), "value must be a 4D tensor: [B, num_kv_heads, seq_len, head_size]."

    assert key.shape == value.shape, "key and value must have the same shape."
    assert (
        key.shape[-1] == query.shape[-1]
    ), "key and query must have the same head size."

    assert (
        query.shape[1] % key.shape[1] == 0
    ), "num_heads must be divisible by num_kv_heads."

    assert query.shape[2] % 32 == 0, "query sequence length must be divisible by 32."

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
        batch_size = query.shape[1]
        num_heads = query.shape[2]
        head_size = query.shape[3]
        max_seq_len = key.shape[-2]
        query = query.reshape(batch_size, num_heads, 1, head_size)
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(batch_size, num_heads, 1, max_seq_len)
        else:
            attn_mask = torch.zeros(
                batch_size, num_heads, 1, max_seq_len, dtype=query.dtype
            )
            for batch_idx in range(batch_size):
                attn_mask[batch_idx, ..., cur_pos_tensor[batch_idx] + 1 :] = float(
                    "-inf"
                )

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
    attn_mask: torch.Tensor = None,
    cur_pos_tensor: torch.Tensor = None,
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


# Allow the torch dynamo to trace our custom operation(s). This will allow
# the tt custom operation(s) to be represented in a torch.fx.GraphModule.
torch._dynamo.allow_in_graph(torch.ops.tt.mark_argument_attributes)
torch._dynamo.allow_in_graph(torch.ops.tt.update_cache)
torch._dynamo.allow_in_graph(torch.ops.tt.fill_cache)
torch._dynamo.allow_in_graph(torch.ops.tt.scaled_dot_product_attention_decode)
