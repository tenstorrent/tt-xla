# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Tuple

import torch
import torch_xla
from torch._dynamo import register_backend
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind

from .decompositions import CUSTOM_DECOMPOSITION_TABLE
from .metadata_propagation import MetadataDispatchMode, extract_nodes_info
from .passes import (
    bypass_assert_tensor_metadata,
    bypass_dtype_promotion_and_redundant_cast,
    bypass_redundant_getitem,
    handle_composite_ops,
    insert_argument_type_markers,
)


def tt_torch_helper(model, fake_tensor_inputs):
    import torch_xla.core.dynamo_bridge as bridge
    from functorch.compile import make_boxed_func

    compiled_graph = None
    node_info = None

    def fwd(*args):
        nonlocal model
        nonlocal compiled_graph
        nonlocal node_info
        if compiled_graph is None:
            model = torch_pass_pipeline(model, args)
            node_info = extract_nodes_info(model)
            with MetadataDispatchMode(node_info):
                compiled_graph = bridge.extract_compiled_graph(model, args)
            del model
        return compiled_graph(*args)

    return make_boxed_func(fwd)


# This function runs a series of passes on a torch GraphModule.
# The passes here may be necessary (depending on the model) to
# convert a GraphModule into a form which tt-mlir can compile/execute.
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
) -> Tuple[torch.fx.GraphModule, torch.export.ExportGraphSignature, list[str]]:

    # Currently, handle_composite_ops causes regressions on multi-chip TP models:
    # https://github.com/tenstorrent/tt-xla/issues/1616.
    # TODO: Fix composite ops to support multi-chip models before uncommenting this.
    # handle_composite_ops(gm)

    # decompositions = torch._decomp.core_aten_decompositions()
    # decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
    #
    # We use `export_for_training` here as we plan to use this flow to compile training graphs.
    # In addition to that, the functionality in `export_for_training` will become the default
    # functionality in torch.export in a future PyTorch release:
    # https://docs.pytorch.org/docs/stable/export.html#export-for-training-and-inference
    compiled_graph = bypass_dtype_promotion_and_redundant_cast(gm, example_inputs)
    compiled_graph = bypass_redundant_getitem(compiled_graph)
    compiled_graph = bypass_assert_tensor_metadata(compiled_graph)

    # Recompile the GraphModule to ensure the modifications made by the above
    # passes are reflected during execution.
    compiled_graph.recompile()

    # Extract metadata from FX nodes in order to inject them into locs
    # node_info = extract_nodes_info(compiled_graph)

    return compiled_graph


from torch._dynamo.backends.torchxla import openxla


class XLAExecutor:
    """
    This class is used to execute a compiled program on an XLA device.
    It is responsible for:
    1. Executing the GraphModule
    2. Signalling to torch-xla to cut the graph at the model output.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        signature: torch.export.ExportGraphSignature,
        node_info: list[str],
    ):
        self.module = module
        self.signature = signature
        self.node_info = node_info
        # Inject metadata if xla debug is enabled and node_info is not empty
        # We need xla debug to be enabled in order for torch-xla to inject metadata
        self.inject_metadata = os.environ.get("XLA_HLO_DEBUG", "0") == "1" and node_info

        # Collect all devices this model will use. This device list is used to
        # signal to torch xla which devices are involved in computing the output
        # tensors, so that we may cut the graph on the output tensors correctly.
        self.devices = set()
        for _, tensor in module.state_dict().items():
            self.devices.add(tensor.device.type)
        self.devices = list(self.devices)

    def __call__(self, *args):

        if self.inject_metadata:
            # MetadataDispatchMode intercepts tensor operations via TorchDispatchMode and
            # attaches FX metadata (module hierarchy, file, line) to XLA tensors.
            with MetadataDispatchMode(self.node_info):
                output = self.module(*args)
        else:
            output = self.module(*args)

        gm_has_functional_output_kind: bool = True

        for el in self.signature.output_specs:
            if el.kind is not OutputKind.USER_OUTPUT:
                gm_has_functional_output_kind = False
                break

        if gm_has_functional_output_kind:
            # This tells torch-xla to cut the graph at only what is required to
            # compute all tensors in the `output` list.
            torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
        else:
            # Some graphs have side effects not included in graph output.
            # In these cases we must call sync() to force materialization of non-user-output
            # tensors, eg. inplace static cache updates as OutputKind.USER_INPUT_MUTATION.
            # This causes buffer mutations to show up as graph outputs in MLIR.
            torch_xla.sync()

        return output


class TTBackendCompiler:
    def __init__(self, **kwargs):
        from torch._dynamo.backends.common import aot_autograd

        decompositions = torch._decomp.core_aten_decompositions()
        decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
        self.aot_autograd = aot_autograd(
            fw_compiler=tt_torch_helper,
            decompositions=decompositions,
        )

    def __call__(self, gm: torch.fx.GraphModule, example_inputs, **kwargs):
        program = torch.export.export_for_training(
            gm, tuple(example_inputs), strict=False
        )
        gm = insert_argument_type_markers(program.module(), program.graph_signature)

        return self.aot_autograd(gm, example_inputs, **kwargs)


def create_compiler_fn() -> TTBackendCompiler:
    return TTBackendCompiler()


assert callable(create_compiler_fn())

register_backend(name="tt", compiler_fn=create_compiler_fn())

# def xla_backend():
#     """TT backend for torch.compile."""
#     from torch._dynamo.backends.common import aot_autograd
#     decompositions = torch._decomp.core_aten_decompositions()
#     decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
#     return aot_autograd(
#         fw_compiler=tt_torch_helper,
#         decompositions=decompositions,
#     )
#     # module, graph_signature, node_info = torch_pass_pipeline(gm, example_inputs)
#     # return XLAExecutor(module, graph_signature, node_info)
#
# register_backend(name="tt", compiler_fn=xla_backend)
