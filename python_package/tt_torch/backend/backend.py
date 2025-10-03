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
    bypass_dtype_promotion_and_redundant_cast,
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
    return compiled_graph


class XLAExecutor:
    """
    This class is used to execute a compiled program on an XLA device.
    It is responsible for:
    1. Executing the GraphModule
    2. Signalling to torch-xla to cut the graph at the model output.
    """

    def __init__(self, module: torch.fx.GraphModule, auto_to_xla: bool = False):
        self.module = module

        # Collect all devices this model will use. This device list is used to
        # signal to torch xla which devices are involved in computing the output
        # tensors, so that we may cut the graph on the output tensors correctly.
        self.devices = set()
        self.auto_to_xla = auto_to_xla
        if self.auto_to_xla:
            self.module = self.module.to("xla")
        for _, tensor in module.state_dict().items():
            if hasattr(tensor, "device"):
                self.devices.add(tensor.device.type)
        self.devices = list(self.devices)

    def __call__(self, *args):
        args = list(args)
        for idx, arg in enumerate(args):
            if self.auto_to_xla and arg.device.type != "xla":
                args[idx] = arg.to("xla")
        args = tuple(args)
        output = self.module(*args)
        # This tells torch-xla to cut the graph at only what is required to
        # compute all tensors in the `output` list.
        torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
        output = list(output)
        for idx, out in enumerate(output):
            if hasattr(out, "to"):
                output[idx] = out.to("cpu")
        output = tuple(output)
        assert False
        return output


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options={}):
    auto_to_xla = options.get("auto_to_xla", True)

    module = torch_pass_pipeline(gm, example_inputs)
    return XLAExecutor(module, auto_to_xla)
