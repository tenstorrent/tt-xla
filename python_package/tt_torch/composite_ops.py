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


def composite_topk(
    input: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    out: tuple[Tensor, ...] | list[Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Creates composite topk operation for torch-xla using StableHLOCompositeBuilder.
    Returns a (values, indices) tuple.
    """
    attrs = {"k": k, "dim": dim, "largest": largest, "sorted": sorted}

    builder = StableHLOCompositeBuilder(name="tenstorrent.topk", attr=attrs)

    input = builder.mark_inputs(input)
    values, indices = torch.topk(input, k, dim, largest, sorted, out=out)
    values, indices = builder.mark_outputs(values, indices)
    return (values, indices)


def composite_topk_values(
    input: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    out: tuple[Tensor, ...] | list[Tensor] | None = None,
) -> Tensor:
    """Composite topk returning only values. Marks a single output at pos=0."""
    attrs = {"k": k, "dim": dim, "largest": largest, "sorted": sorted}
    builder = StableHLOCompositeBuilder(name="tenstorrent.topk_values", attr=attrs)
    input = builder.mark_inputs(input)
    values, _ = torch.topk(input, k, dim, largest, sorted)
    values = builder.mark_outputs(values)
    return values


def composite_topk_indices(
    input: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    out: tuple[Tensor, ...] | list[Tensor] | None = None,
) -> Tensor:
    """Composite topk returning only indices. Marks a single output at pos=0."""
    attrs = {"k": k, "dim": dim, "largest": largest, "sorted": sorted}
    builder = StableHLOCompositeBuilder(name="tenstorrent.topk_indices", attr=attrs)
    input = builder.mark_inputs(input)
    _, indices = torch.topk(input, k, dim, largest, sorted)
    indices = builder.mark_outputs(indices)
    return indices


def composite_group_norm(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Creates composite group_norm operation for torch xla using StableHLOCompositeBuilder.
    Operation name is tenstorrent.group_norm for MLIR to handle it.

    Args:
        input: Input tensor to normalize
        num_groups: Number of groups to divide the channels into
        weight: Optional learnable weight parameter for affine transformation
        bias: Optional learnable bias parameter for affine transformation
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized tensor with same shape as input
    """
    attr = {"num_groups": num_groups, "epsilon": eps, "channel_dim": 1}

    builder = StableHLOCompositeBuilder(name="tenstorrent.group_norm", attr=attr)

    if weight is not None and bias is not None:
        input, weight, bias = builder.mark_inputs(input, weight, bias)
    elif weight is not None:
        input, weight = builder.mark_inputs(input, weight)
    elif bias is not None:
        input, bias = builder.mark_inputs(input, bias)
    else:
        input = builder.mark_inputs(input)

    output = torch.nn.functional.group_norm(input, num_groups, weight, bias, eps)

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
        # Copy metadata from original node to preserve stack_trace and nn_module_stack
        # This ensures layer_norm maintains proper source location in MLIR output
        new_node.meta = node.meta.copy()

    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


def replace_group_norm_module(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    module: torch.nn.GroupNorm,
) -> None:
    """
    Replace nn.GroupNorm call_module node with composite_group_norm call_function.

    Transformation:
        BEFORE: %out = call_module[target=group_norm](args=(%x,))
        AFTER:  %weight = get_attr[target=group_norm.weight]
                %bias = get_attr[target=group_norm.bias]
                %out = call_function[target=composite_group_norm](
                    args=(%x,),
                    kwargs={num_groups: 8, weight: %weight,
                            bias: %bias, eps: 1e-5}
                )

    Args:
        gm: GraphModule containing the node
        node: call_module node to replace
        module: nn.GroupNorm instance
    """
    num_groups = module.num_groups
    eps = module.eps
    has_weight = module.weight is not None and module.affine
    has_bias = module.bias is not None and module.affine

    input_tensor = node.args[0]

    kwargs = {"num_groups": num_groups, "eps": eps}

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
            composite_group_norm, args=(input_tensor,), kwargs=kwargs
        )

    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


"""
Dictionary holding replacement composite functions for torch functions and modules.
Maps torch API calls and module types to composite implementations.
Used for call_function and call_module nodes where node.target is a function reference or module type.

When the node.target is a torch function returning just a single output, we use the
composite function directly.

When the node.target is a torch function returning multiple outputs, we use a dictionary
of composite functions, keyed by the frozenset of output indices used.
"""
replacements = {
    # function replacements
    torch.nn.functional.gelu: composite_gelu,
    torch.rms_norm: composite_rms_norm,
    torch.nn.functional.rms_norm: composite_rms_norm,
    torch.nn.functional.layer_norm: composite_layer_norm,
    torch.topk: {
        frozenset({0, 1}): composite_topk,
        frozenset({0}): composite_topk_values,
        frozenset({1}): composite_topk_indices,
    },
    torch.nn.functional.group_norm: composite_group_norm,
    # module replacements
    torch.nn.LayerNorm: replace_layer_norm_module,
    torch.nn.GroupNorm: replace_group_norm_module,
}

"""
Maps tensor method name strings to their torch function equivalents.
Used to resolve call_method FX nodes (e.g. x.topk(k)) where dynamo sets
node.target to the method name string "topk".
The mapped function must be a key in `replacements` for the rewrite to apply.
"""
method_name_to_function = {
    "topk": torch.topk,
}
