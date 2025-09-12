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
