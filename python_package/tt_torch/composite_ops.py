# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder
import torch.nn.functional as F

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
    input = F.gelu(input, approximate=approximate)
    input = builder.mark_outputs(input)

    return input


"""
Dictionary holding replacement composite functions for torch functions.
"""
replacements = {torch.nn.functional.gelu: composite_gelu}


def handle_composite_ops(gm: torch.fx.GraphModule) -> None:
    """
    Replaces torch ops with composite ops if we have a proper replacement.

    Note that there is no need to recompile graph here since it will be recompiled after
    all manipulations are done in python_package/tt_torch/backend/backend.py.
    """
    for node in gm.graph.nodes:
        if node.target in replacements:
            node.target = replacements[node.target]
