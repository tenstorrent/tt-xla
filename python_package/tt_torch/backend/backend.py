# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from types import DynamicClassAttribute
from typing import Tuple, Optional
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
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import numpy as np
from torch_xla.distributed.spmd import Mesh, ShardingType


class DynamoBackendOptions:
    def __init__(self, mesh: Optional[Mesh]):
        self.mesh = mesh
        self.remove_dummy_io = False


# This function inspects all tensors in the state_dict of the given GraphModule,
# as well as the provided example inputs, to determine if any of them have a sharding
# annotation that is not fully replicated.
def has_sharded_inputs(
    gm: torch.fx.GraphModule, example_inputs: Tuple[torch.Tensor]
) -> bool:
    tensors = list(gm.state_dict().values())
    tensors.extend(example_inputs)
    for tensor in tensors:
        if (
            torch_xla._XLAC._get_xla_sharding_type(tensor) is not None
            and torch_xla._XLAC._get_xla_sharding_type(tensor)
            != ShardingType.REPLICATED
        ):
            return True
    return False


# Add a dummy srarded input and output to the graph. This is needed because currently, fully sharded graphs
# are not represented correctly by torch-xla and we get a graph without a mesh op. This is a temporary
# workaround until https://github.com/tenstorrent/tt-xla/issues/1487 is resolved.
def add_dummy_sharded_io(
    gm: torch.fx.GraphModule, options: DynamoBackendOptions, device: torch.device
) -> torch.fx.GraphModule:
    dummy_io = torch.randn(
        32 * options.mesh.mesh_shape[0],
        32 * options.mesh.mesh_shape[1],
        dtype=torch.bfloat16,
    )
    gm.register_parameter("dummy_io", torch.nn.Parameter(dummy_io))
    gm = gm.to(device)
    xs.mark_sharding(
        gm.dummy_io,
        options.mesh,
        (options.mesh.axis_names[0], options.mesh.axis_names[1]),
    )
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            with gm.graph.inserting_before(node):
                input_node = gm.graph.get_attr("dummy_io")
        if node.op == "output":
            with gm.graph.inserting_before(node):
                current_outputs = list(node.args[0])
                node.args = (tuple(current_outputs + [input_node]),)
            break

    options.remove_dummy_io = True
    return gm


# This function runs a series of passes on a torch GraphModule.
# The passes here may be necessary (depending on the model) to
# convert a GraphModule into a form which tt-mlir can compile/execute.
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: DynamoBackendOptions,
) -> torch.fx.GraphModule:
    decompositions = torch._decomp.core_aten_decompositions()
    decompositions.update(CUSTOM_DECOMPOSITION_TABLE)

    # dummy input needed for replicated inputs in SPMD mode.See comment in add_dummy_sharded_io
    num_devices = options.mesh.mesh_shape[0] * options.mesh.mesh_shape[1]
    if num_devices > 1 and not has_sharded_inputs(gm, example_inputs):
        gm = add_dummy_sharded_io(gm, options, example_inputs[0].device)

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

    def __init__(self, module: torch.fx.GraphModule, options: DynamoBackendOptions):
        self.module = module
        self.options = options

        # Collect all devices this model will use. This device list is used to
        # signal to torch xla which devices are involved in computing the output
        # tensors, so that we may cut the graph on the output tensors correctly.
        self.devices = set()
        for _, tensor in module.state_dict().items():
            self.devices.add(tensor.device.type)
        self.devices = list(self.devices)

    def __call__(self, *args):

        output = self.module(*args)

        torch_xla.sync()
        if self.options.remove_dummy_io:
            output = output[:-1]


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options=DynamoBackendOptions(None)):
    if options.mesh is None:
        options.mesh = xs.get_global_mesh()
        assert options.mesh is not None

    module = torch_pass_pipeline(gm, example_inputs, options)
    return XLAExecutor(module, options)
