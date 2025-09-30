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


global_counter = 0

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
    compiled_graph = bypass_dtype_promotion(compiled_graph)
    compiled_graph = bypass_redundant_cast(compiled_graph)
    compiled_graph = bypass_redundant_getitem(compiled_graph)
    compiled_graph = bypass_assert_tensor_metadata(compiled_graph)

    # Recompile the GraphModule to ensure the modifications made by the above
    # passes are reflected during execution.
    compiled_graph.recompile()

    # Inject semantic location metadata into FX nodes for torch-xla
    _inject_semantic_locations(compiled_graph)

    return compiled_graph


def _inject_semantic_locations(graph_module):
    """Extract and inject semantic location metadata into FX nodes."""
    location_mapping = {}

    for node in graph_module.graph.nodes:
        if not hasattr(node, 'meta') or not node.meta:
            continue

        location_info = {}

        # Extract file and line from stack_trace
        if 'stack_trace' in node.meta and node.meta['stack_trace']:
            for line in node.meta['stack_trace'].strip().split('\n'):
                if 'File "' in line and ', line ' in line:
                    parts = line.strip().split('"')
                    if len(parts) >= 2:
                        location_info['file'] = parts[1]
                        location_info['line'] = int(line.split(', line ')[1].split(',')[0])
                        break

        # Extract module hierarchy from nn_module_stack
        if 'nn_module_stack' in node.meta and node.meta['nn_module_stack']:
            modules = sorted(node.meta['nn_module_stack'].items(), key=lambda x: len(x[1][0]))
            module_classes = []
            for _, (_, class_name) in modules:
                clean = class_name.split('.')[-1] if '.' in class_name else class_name
                if clean not in module_classes:
                    module_classes.append(clean)
            if module_classes:
                location_info['module_path'] = '/'.join(module_classes)

        # Extract operation name
        if 'torch_fn' in node.meta and node.meta['torch_fn'] and len(node.meta['torch_fn']) >= 2:
            location_info['op_name'] = node.meta['torch_fn'][0]

        # Create and inject semantic location
        if location_info and 'op_name' in location_info:
            file_name = location_info.get('file', 'unknown').split('/')[-1]
            line_num = location_info.get('line', 0)
            module_path = location_info.get('module_path', '')

            # Build hierarchical location string
            if module_path:
                modules = module_path.split('/')
                path_parts = []
                for mod in modules[:-1]:
                    path_parts.extend([mod, 'forward'])
                path_parts.extend([modules[-1], 'forward', node.name])
                hierarchy = '->'.join(path_parts)
            else:
                hierarchy = f"forward->{node.name}"

            semantic_loc = f"{hierarchy}({file_name}:{line_num})"
            node.meta['tt_semantic_location'] = semantic_loc
            location_mapping[location_info['op_name']] = semantic_loc

    graph_module._tt_location_mapping = location_mapping


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

        # Store location mapping for semantic interpreter
        self.location_mapping = getattr(module, '_tt_location_mapping', {})

    def __call__(self, *args):
        # Enable FX context and use semantic interpreter if we have location mapping
        if self.location_mapping:
            torch_xla._XLAC._set_thread_local_fx_context(True)
            try:
                output = self._execute_with_semantic_interpreter(*args)
            finally:
                torch_xla._XLAC._set_thread_local_fx_context(False)
        else:
            output = self.module(*args)

        # This tells torch-xla to cut the graph at only what is required to
        # compute all tensors in the `output` list.
        torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
        return output

    def _execute_with_semantic_interpreter(self, *args):
        """Execute graph with semantic location tracking per FX node."""
        class SemanticInterpreter(torch.fx.Interpreter):
            def __init__(self, module, location_mapping):
                super().__init__(module)
                self.location_mapping = location_mapping

            def run_node(self, node):
                global global_counter

                # Set semantic location before executing node
                semantic_loc = node.meta.get('tt_semantic_location', '')
                print("--- Global counter: ", global_counter)
                global_counter += 1
                print("--- Node in semantic interpreter: ", node)
                print("--- semantic location: ", semantic_loc)
                print("================================================")
                if semantic_loc:
                    torch_xla._XLAC._set_current_fx_semantic_location(semantic_loc)
                try:
                    result = super().run_node(node)
                finally:
                    if semantic_loc:
                        torch_xla._XLAC._clear_current_fx_semantic_location()
                return result

        interpreter = SemanticInterpreter(self.module, self.location_mapping)
        return interpreter.run(*args)


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options=None):

    module = torch_pass_pipeline(gm, example_inputs)
    return XLAExecutor(module)
