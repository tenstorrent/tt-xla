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


# Allow the torch dynamo to trace our custom operation(s). This will allow
# the tt custom operation(s) to be represented in a torch.fx.GraphModule.
torch._dynamo.allow_in_graph(torch.ops.tt.mark_argument_attributes)
