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

    return stablehlo_custom_call.stablehlo_custom_call(
        [tensor],
        "tt.mark_argument",
        [tensor.shape],
        [tensor.dtype],
        frontend_attributes=frontend_attributes,
    )


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
