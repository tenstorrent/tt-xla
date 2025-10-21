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
) -> Tuple[torch.export.ExportedProgram, torch.export.ExportGraphSignature]:

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

    program = torch.export.export_for_training(
        compiled_graph, tuple(example_inputs), strict=False
    )
    return program, program.graph_signature


from collections import defaultdict


def build_full_args_for_gm(ep: ExportedProgram, *user_args):
    gm = ep.graph_module
    sig = ep.graph_signature

    # Export keeps a state dict for lifted params/buffers
    state = ep.state_dict  # { "p_...": tensor, "b_...": tensor, ... }
    for k, v in state.items():
        print(f"state: {k}")

    # Some exports also have constant tensors; include them if present
    constants = getattr(ep, "constants", {})  # may be {} on your version

    # Map from placeholder name -> tensor
    lookup = defaultdict(lambda: None)
    total_args = tuple()
    for spec in sig.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            continue
        total_args += (state[spec.target],)

    return total_args
    # Params
    for spec in sig.parameters:
        print(f"parameter: {spec}")
        lookup[spec] = state[spec]
    # Buffers
    for spec in getattr(sig, "buffers", []):
        print(f"buffer: {spec}")
        lookup[spec] = state[spec]
    # Constant tensors (if any)
    for spec in getattr(sig, "constants", []):
        print(f"const: {spec}")
        lookup[spec] = constants[spec]

    print(f"code:\n{gm.code}")
    print(f"gm.buffers(): {[name for (name, _) in gm.named_buffers()]}")

    for name, buffer in ep.named_buffers():
        print(f"buffer: {name}")
        lookup[name] = buffer
    # Now assemble args in the exact placeholder order
    full_args = []
    user_it = iter(user_args)
    for n in gm.graph.nodes:
        if n.op != "placeholder":
            continue
        name = n.target
        if name in lookup and lookup[name] is not None:
            full_args.append(lookup[name])  # lifted thing
        else:
            print(f"Using user arg for placeholder '{name}'")
            full_args.append(next(user_it))  # user input

    return tuple(full_args)


class XLAExecutor:
    """
    This class is used to execute a compiled program on an XLA device.
    It is responsible for:
    1. Executing the GraphModule
    2. Signalling to torch-xla to cut the graph at the model output.
    """

    def __init__(
        self,
        module: torch.export.ExportedProgram,
        signature: torch.export.ExportGraphSignature,
    ):
        self.module = module
        self.signature = signature
        self.compiled_graph = None
        self.full_args = None

        # Collect all devices this model will use. This device list is used to
        # signal to torch xla which devices are involved in computing the output
        # tensors, so that we may cut the graph on the output tensors correctly.
        self.module.devices = set()
        for _, tensor in module.graph_module.state_dict().items():
            self.module.devices.add(tensor.device.type)
        self.devices = list(self.module.devices)

    def __call__(self, *args):

        if self.compiled_graph is None:
            import torch_xla.core.dynamo_bridge as bridge

            self.full_args = build_full_args_for_gm(self.module, *args)
            self.compiled_graph = bridge.extract_compiled_graph(
                self.module.graph_module, self.full_args + args
            )

        output = self.compiled_graph(*(self.full_args + args))

        gm_has_functional_output_kind: bool = True

        for el in self.signature.output_specs:
            if el.kind is not OutputKind.USER_OUTPUT:
                gm_has_functional_output_kind = False
                break

        # if gm_has_functional_output_kind:
        #     # This tells torch-xla to cut the graph at only what is required to
        #     # compute all tensors in the `output` list.
        #     torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
        # else:
        #     # Some graphs have side effects not included in graph output.
        #     # In these cases we must call sync() to force materialization of non-user-output
        #     # tensors, eg. inplace static cache updates as OutputKind.USER_INPUT_MUTATION.
        #     # This causes buffer mutations to show up as graph outputs in MLIR.
        #     torch_xla.sync()
        #
        return output


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options=None):

    module, graph_signature = torch_pass_pipeline(gm, example_inputs)
    return XLAExecutor(module, graph_signature)
