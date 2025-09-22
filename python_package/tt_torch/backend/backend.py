# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Tuple

import torch
import torch_xla
from torch._dynamo import register_backend
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

from .decompositions import CUSTOM_DECOMPOSITION_TABLE
from .passes import (
    bypass_assert_tensor_metadata,
    bypass_dtype_promotion_and_redundant_cast,
    bypass_redundant_getitem,
    handle_composite_ops,
    insert_argument_type_markers,
)


# This function runs a series of passes on a torch GraphModule.
# The passes here may be necessary (depending on the model) to
# convert a GraphModule into a form which tt-mlir can compile/execute.
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
) -> torch.fx.GraphModule:

    # Currently, handle_composite_ops causes regressions on multi-chip TP models:
    # https://github.com/tenstorrent/tt-xla/issues/1616.
    # TODO: Fix composite ops to support multi-chip models before uncommenting this.
    # handle_composite_ops(gm)

    decompositions = torch._decomp.core_aten_decompositions()
    decompositions.update(CUSTOM_DECOMPOSITION_TABLE)

    # We use `export_for_training` here as we plan to use this flow to compile training graphs.
    # In addition to that, the functionality in `export_for_training` will become the default
    # functionality in torch.export in a future PyTorch release:
    # https://docs.pytorch.org/docs/stable/export.html#export-for-training-and-inference
    # print("example_inputs", example_inputs)
    # print("gm state_dict keys:", list(gm.state_dict().keys()))
    print("[james] override use torch.export.export")
    program = torch.export.export_for_training(
        gm, tuple(example_inputs), strict=False
    ).run_decompositions(decompositions)
    print("program.graph_signature:", program.graph_signature)
    # print("program inputs:", [str(inp) for inp in program.graph_signature.input_specs])

    compiled_graph = program.module()
    compiled_graph = insert_argument_type_markers(
        compiled_graph, program.graph_signature
    )
    compiled_graph = bypass_dtype_promotion_and_redundant_cast(
        compiled_graph, example_inputs
    )
    compiled_graph = bypass_redundant_getitem(compiled_graph)
    compiled_graph = bypass_assert_tensor_metadata(compiled_graph)

    # Recompile the GraphModule to ensure the modifications made by the above
    # passes are reflected during execution.
    compiled_graph.recompile()

    # print("post pass program.graph_signature:", program.graph_signature)
    # print("post pass program inputs:", [str(inp) for inp in program.graph_signature.input_specs])

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

        # print("readable graph module @ executor call")
        # self.module.print_readable()
        output = self.module(*args)

        # This tells torch-xla to cut the graph at only what is required to
        # compute all tensors in the `output` list.
        # torch_xla.sync()
        torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
        return output


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options=None):

    module = torch_pass_pipeline(gm, example_inputs)
    return XLAExecutor(module)
