# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
import torch
from torch.export import ExportedProgram


from .decompositions import (
    CUSTOM_DECOMPOSITION_TABLE,
)
import os
from .passes import (
    bypass_redundant_getitem,
    bypass_dtype_promotion,
    bypass_redundant_cast,
    insert_argument_type_markers,
    bypass_assert_tensor_metadata,
)

from torch.export.graph_signature import InputKind
from torch._dynamo import register_backend

import torch_xla


# This function runs a series of passes on a torch GraphModule.
# The passes here may be necessary (depending on the model) to
# convert a GraphModule into a form which tt-mlir can compile/execute.
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
) -> torch.fx.GraphModule:
    print("\n[DEBUG] torch_pass_pipeline received graph:")
    print("  Nodes in graph:")
    for i, node in enumerate(gm.graph.nodes):
        print(f"    {i}: {node.op} - {node.target}")
        # Check specifically for GELU
        if node.op == "call_function":
            if "gelu" in str(node.target).lower():
                print(f"      ^^^ GELU FOUND! Not decomposed yet!")
            elif "mark_tensor" in str(node.target):
                print(f"      ^^^ mark_tensor from composite builder!")

    decompositions = torch._decomp.core_aten_decompositions()
    decompositions.update(CUSTOM_DECOMPOSITION_TABLE)

    # We use `export_for_training` here as we plan to use this flow to compile training graphs.
    # In addition to that, the functionality in `export_for_training` will become the default
    # functionality in torch.export in a future PyTorch release:
    # https://docs.pytorch.org/docs/stable/export.html#export-for-training-and-inference
    program = torch.export.export_for_training(
        gm, tuple(example_inputs), strict=False
    ).run_decompositions(decompositions)

    compiled_graph = program.module()

    print("\n[DEBUG] After decomposition pass:")
    print("  Nodes in graph:")
    for i, node in enumerate(compiled_graph.graph.nodes):
        # Only show first 10 nodes to keep output manageable
        if i < 10:
            print(f"    {i}: {node.op} - {node.target}")
            if node.op == "call_function":
                if "gelu" in str(node.target).lower():
                    print(f"      ^^^ GELU STILL HERE! NOT decomposed by PyTorch!")
                elif "mark_tensor" in str(node.target):
                    print(f"      ^^^ mark_tensor still present!")
    if len(list(compiled_graph.graph.nodes)) > 10:
        print(f"    ... and {len(list(compiled_graph.graph.nodes)) - 10} more nodes")

    # Note: Temporarily commenting out insert_argument_type_markers when using CPU PJRT
    # The mark_argument custom ops cause errors with CPU PJRT backend
    # TODO: Re-enable when running on actual TT hardware
    # compiled_graph = insert_argument_type_markers(
    #     compiled_graph, program.graph_signature
    # )
    compiled_graph = bypass_dtype_promotion(compiled_graph)
    compiled_graph = bypass_redundant_cast(compiled_graph)
    compiled_graph = bypass_redundant_getitem(compiled_graph)
    compiled_graph = bypass_assert_tensor_metadata(compiled_graph)

    # Recompile the GraphModule to ensure the modifications made by the above
    # passes are reflected during execution.
    compiled_graph.recompile()
    return compiled_graph


class XLAExecutor:
    """
    This class is used to execute a compiled program on an XLA device.
    It is responsible for:
    1. Executing the GraphModule
    2. Signalling to torch-xla to cut the graph at the model output.
    """

    def __init__(self, module: torch.fx.GraphModule):
        self.module = module

        # Collect all devices this model will use. This device list is used to
        # signal to torch xla which devices are involved in computing the output
        # tensors, so that we may cut the graph on the output tensors correctly.
        self.devices = set()
        for _, tensor in module.state_dict().items():
            self.devices.add(tensor.device.type)
        self.devices = list(self.devices)

    def __call__(self, *args):

        output = self.module(*args)
        # This tells torch-xla to cut the graph at only what is required to
        # compute all tensors in the `output` list.
        torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
        return output


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options=None):

    module = torch_pass_pipeline(gm, example_inputs)
    return XLAExecutor(module)
