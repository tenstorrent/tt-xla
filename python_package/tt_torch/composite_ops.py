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

    This function supports both functional API usage and module replacement:
    - Functional: torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-5)
    - Module: nn.LayerNorm(768) → replaced with this composite

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


"""
Dictionary holding replacement composite functions for torch functions.
Maps functional API calls to composite implementations.
Used for call_function nodes where node.target is a function reference.
"""
function_replacements = {
    torch.nn.functional.gelu: composite_gelu,
    torch.rms_norm: composite_rms_norm,
    torch.nn.functional.layer_norm: composite_layer_norm,
}

"""
Maps nn.Module classes to composite implementations.
Used for call_module nodes where we transform module calls to functional calls.
"""
module_replacements = {
    torch.nn.LayerNorm: composite_layer_norm,
}
