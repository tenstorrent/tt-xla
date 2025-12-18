# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder

"""
From XLA documentation: '(Composite operation) Encapsulates an operation made up (composed) of
other StableHLO operations, taking inputs and composite_attributes and producing results.
The semantics of the op are implemented by the decomposition attribute. The composite op can be
replaced with its decomposition without changing program semantics.'

So, composite op is made because of high-level ops that are decomposed into dozens stable HLO
operations. It enables custom backends to handle op however they want. Since we have a native
support for, let's say, gelu operation in ttir, we will wrap torch gelu op into gelu composite
op and handle it in XLA without decompostions, enabling our device to execute custom gelu implementation.

Since we want to run torch models wihout modifying them, we will substitute torch operations
(that we have a direct support for) with composite ops. This way, user will not have to change
anything in model in order to get performance improvement.
"""

################# helper functions #################


def _normalize_tuple_param(param: Union[int, tuple]) -> List[int]:
    """
    Normalize int or tuple parameter to list of 2 ints for 2D operations.

    Args:
        param: Either an int or tuple of 2 ints

    Returns:
        List of 2 ints
    """
    if isinstance(param, int):
        return [param, param]
    else:
        return list(param)


################# function replacements #################


def composite_gelu(input: Tensor, approximate: str = "none") -> Tensor:
    """
    Creates composite gelu operation for torch xla using StableHLOCompositeBuilder.
    Note that operation name must be tenstorrent.gelu[_tanh] for MLIR to handle it.

    Returns a tensor.
    """
    tanh = approximate == "tanh"
    name = "tenstorrent.gelu" + ("_tanh" if tanh else "")
    attr = {"approximate": "tanh"} if tanh else None

    builder = StableHLOCompositeBuilder(name=name, attr=attr)

    input = builder.mark_inputs(input)
    input = torch.nn.functional.gelu(input, approximate=approximate)
    input = builder.mark_outputs(input)

    return input


def composite_rms_norm(
    input: Tensor, normalized_shape, weight=None, eps=None
) -> Tensor:
    """
    Creates composite RMS norm operation for torch xla using StableHLOCompositeBuilder.
    Note that operation name must be tenstorrent.rms_norm for MLIR to handle it.

    Args:
        input: Input tensor
        normalized_shape: Shape over which to normalize (tuple of ints)
        weight: Optional learnable weight parameter
        eps: Epsilon for numerical stability (default: None)

    Returns a tensor.
    """
    attr = {"normalized_shape": normalized_shape}
    if eps is not None:
        attr["epsilon"] = eps

    builder = StableHLOCompositeBuilder(name="tenstorrent.rms_norm", attr=attr)

    if weight is not None:
        input, weight = builder.mark_inputs(input, weight)
    else:
        input = builder.mark_inputs(input)

    output = torch.nn.functional.rms_norm(input, normalized_shape, weight, eps)
    output = builder.mark_outputs(output)

    return output


def composite_layer_norm(
    input: Tensor,
    normalized_shape: Union[int, List[int], torch.Size],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Creates composite layer_norm operation for torch xla using StableHLOCompositeBuilder.
    Operation name is tenstorrent.layer_norm for MLIR to handle it.

    Args:
        input: Input tensor to normalize
        normalized_shape: Shape over which to normalize (int, list, tuple, or torch.Size)
        weight: Optional learnable weight parameter for affine transformation
        bias: Optional learnable bias parameter for affine transformation
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized tensor with same shape as input
    """
    if isinstance(normalized_shape, int):
        normalized_shape_list = [normalized_shape]
    else:
        normalized_shape_list = list(normalized_shape)

    attr = {"normalized_shape": normalized_shape_list, "epsilon": eps}

    builder = StableHLOCompositeBuilder(name="tenstorrent.layer_norm", attr=attr)

    if weight is not None and bias is not None:
        input, weight, bias = builder.mark_inputs(input, weight, bias)
    elif weight is not None:
        input, weight = builder.mark_inputs(input, weight)
    elif bias is not None:
        input, bias = builder.mark_inputs(input, bias)
    else:
        input = builder.mark_inputs(input)

    output = torch.nn.functional.layer_norm(
        input, normalized_shape_list, weight, bias, eps
    )

    output = builder.mark_outputs(output)

    return output


def composite_conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    output_padding: Union[int, tuple] = 0,
    groups: int = 1,
    dilation: Union[int, tuple] = 1,
) -> Tensor:
    """
    Creates composite conv_transpose2d operation for torch xla using StableHLOCompositeBuilder.
    Operation name is tenstorrent.conv_transpose2d for MLIR to handle it.

    Args:
        input: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor
        stride: Stride of the convolution (default: 1)
        padding: Padding added to both sides of the input (default: 0)
        output_padding: Additional size added to one side of the output shape (default: 0)
        groups: Number of blocked connections from input channels to output channels (default: 1)
        dilation: Spacing between kernel elements (default: 1)

    Returns:
        Output tensor after transposed convolution
    """
    # Normalize tuple parameters to lists for StableHLO compatibility
    stride_list = _normalize_tuple_param(stride)
    padding_list = _normalize_tuple_param(padding)
    output_padding_list = _normalize_tuple_param(output_padding)
    dilation_list = _normalize_tuple_param(dilation)

    attr = {
        "stride": stride_list,
        "padding": padding_list,
        "output_padding": output_padding_list,
        "groups": groups,
        "dilation": dilation_list,
    }

    builder = StableHLOCompositeBuilder(name="tenstorrent.conv_transpose2d", attr=attr)

    if bias is not None:
        input, weight, bias = builder.mark_inputs(input, weight, bias)
    else:
        input, weight = builder.mark_inputs(input, weight)

    output = torch.nn.functional.conv_transpose2d(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    output = builder.mark_outputs(output)

    return output


################# module replacements #################


def replace_layer_norm_module(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    module: torch.nn.LayerNorm,
) -> None:
    """
    Replace nn.LayerNorm call_module node with composite_layer_norm call_function.

    Transformation:
        BEFORE: %out = call_module[target=layer_norm](args=(%x,))
        AFTER:  %weight = get_attr[target=layer_norm.weight]
                %bias = get_attr[target=layer_norm.bias]
                %out = call_function[target=composite_layer_norm](
                    args=(%x,),
                    kwargs={normalized_shape: (768,), weight: %weight,
                            bias: %bias, eps: 1e-5}
                )

    Args:
        gm: GraphModule containing the node
        node: call_module node to replace
        module: nn.LayerNorm instance
    """
    normalized_shape = module.normalized_shape
    eps = module.eps
    has_weight = module.weight is not None and module.elementwise_affine
    has_bias = module.bias is not None and module.elementwise_affine

    input_tensor = node.args[0]

    kwargs = {"normalized_shape": normalized_shape, "eps": eps}

    with gm.graph.inserting_before(node):
        if has_weight:
            weight_node = gm.graph.get_attr(f"{node.target}.weight")
            kwargs["weight"] = weight_node
        else:
            kwargs["weight"] = None

        if has_bias:
            bias_node = gm.graph.get_attr(f"{node.target}.bias")
            kwargs["bias"] = bias_node
        else:
            kwargs["bias"] = None

        new_node = gm.graph.call_function(
            composite_layer_norm, args=(input_tensor,), kwargs=kwargs
        )

    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


def replace_conv_transpose2d_module(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    module: torch.nn.ConvTranspose2d,
) -> None:
    """
    Replace nn.ConvTranspose2d call_module node with composite_conv_transpose2d call_function.

    Transformation:
        BEFORE: %out = call_module[target=upsample](args=(%x,))
        AFTER:  %weight = get_attr[target=upsample.weight]
                %bias = get_attr[target=upsample.bias]  # if bias exists
                %out = call_function[target=composite_conv_transpose2d](
                    args=(%x, %weight),
                    kwargs={bias: %bias, stride: (2, 2), padding: (0, 0), ...}
                )

    Args:
        gm: GraphModule containing the node
        node: call_module node to replace
        module: nn.ConvTranspose2d instance
    """
    stride = module.stride
    padding = module.padding
    output_padding = module.output_padding
    groups = module.groups
    dilation = module.dilation
    has_bias = module.bias is not None

    input_tensor = node.args[0]

    kwargs = {
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation,
    }

    with gm.graph.inserting_before(node):
        weight_node = gm.graph.get_attr(f"{node.target}.weight")
        if has_bias:
            bias_node = gm.graph.get_attr(f"{node.target}.bias")
            kwargs["bias"] = bias_node
        else:
            kwargs["bias"] = None

        new_node = gm.graph.call_function(
            composite_conv_transpose2d,
            args=(input_tensor, weight_node),
            kwargs=kwargs,
        )

    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


"""
Dictionary holding replacement composite functions for torch functions and modules.
Maps torch API calls and module types to composite implementations.
Used for call_function and call_module nodes where node.target is a function reference or module type.
"""
replacements = {
    # function replacements
    torch.nn.functional.gelu: composite_gelu,
    torch.rms_norm: composite_rms_norm,
    torch.nn.functional.layer_norm: composite_layer_norm,
    torch.nn.functional.conv_transpose2d: composite_conv_transpose2d,
    # module replacements
    torch.nn.LayerNorm: replace_layer_norm_module,
    torch.nn.ConvTranspose2d: replace_conv_transpose2d_module,
}
