# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Callable

import torch
from torch.fx import GraphModule
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder


def create_composite_wrap_replacement(
    pattern_fn: Callable,
    composite_name: str,
    attr: dict | None = None,
    example_inputs: tuple | None = None,
) -> GraphModule:
    """Creates a traced replacement GraphModule that wraps pattern in composite.

    This function generates a replacement that:
    1. Marks the inputs with StableHLOCompositeBuilder
    2. Executes the original pattern function
    3. Marks the outputs with StableHLOCompositeBuilder

    The replacement is traced using make_fx to produce a GraphModule that
    can be used with torch.fx.subgraph_rewriter.replace_pattern.

    Args:
        pattern_fn: The pattern function to wrap
        composite_name: Name for the StableHLO composite (e.g., 'tenstorrent.op, must match the name in tt-mlir')
        attr: Optional attributes dict for the composite (e.g., {'normalized_shape': (1, 32, 32)})
        example_inputs: Tuple of example tensors for tracing. If None, creates
            dummy tensors based on the pattern signature (shape [1, 32, 32]).

    Returns:
        A traced GraphModule that wraps pattern_fn in composite markers
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    sig = inspect.signature(pattern_fn)
    param_count = len(sig.parameters)

    def replacement(*args):
        builder = StableHLOCompositeBuilder(name=composite_name, attr=attr)
        marked_args = builder.mark_inputs(*args)
        # Handle single arg case (mark_inputs returns single tensor, not tuple)
        if param_count == 1:
            marked_args = (marked_args,)
        result = pattern_fn(*marked_args)
        return builder.mark_outputs(result)

    # Create dummy inputs if not provided
    if example_inputs is None:
        example_inputs = tuple(torch.randn(1, 32, 32) for _ in range(param_count))

    return make_fx(replacement)(*example_inputs)
