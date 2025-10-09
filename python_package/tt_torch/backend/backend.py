# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
import traceback
import torch
from torch.export import ExportedProgram
from torch.utils._python_dispatch import TorchDispatchMode

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

    # Debug: print FX graph if requested
    if os.environ.get("TT_DEBUG_FX_GRAPH", "1") == "1":
        print("\n=== FX Graph Nodes ===")
        for i, node in enumerate(compiled_graph.graph.nodes):
            semantic_loc = node.meta.get('tt_semantic_location', 'NO LOCATION')
            target = node.target if node.op == 'call_function' else ''
            print(f"{i}: {node.op:15s} {node.name:30s} {str(target):40s} -> {semantic_loc}")
        print("=====================\n")

    return compiled_graph


def _clean_fx_name(fqn):
    """
    Clean torch.fx internal naming to get readable instance names.

    torch.fx uses internal prefixes like 'L__self__' in FQNs.
    Examples:
        'L__self__L__self___inner' -> 'inner'
        'L__self__L__self___inner_linear' -> 'linear'
        'inner.linear' -> 'linear'
    """
    if not fqn:
        return ""

    # Remove all 'L__self__' prefixes
    cleaned = fqn
    while 'L__self__' in cleaned:
        cleaned = cleaned.replace('L__self__', '')

    # Strip leading underscores
    cleaned = cleaned.lstrip('_')

    # Take the last component after '.'
    if '.' in cleaned:
        cleaned = cleaned.split('.')[-1]

    return cleaned


def _inject_semantic_locations(graph_module):
    """
    Extract and inject semantic location metadata into FX nodes.

    Format: "ModuleClass[instance]/SubModule[instance]/forward(file.py:line)/"
    Example: "InnerModule[inner]/Linear[linear]/forward(test_nested_model.py:26)/"

    This prefix will be concatenated with the operation type by torch-xla:
    "InnerModule[inner]/Linear[linear]/forward(test_nested_model.py:26)/" + "aten__mm"
    = "InnerModule[inner]/Linear[linear]/forward(test_nested_model.py:26)/aten__mm"
    """
    location_mapping = {}

    for node in graph_module.graph.nodes:
        if not hasattr(node, 'meta') or not node.meta:
            continue

        location_info = {}

        # Extract file and line from stack_trace
        file_name = "unknown"
        line_num = 0
        if 'stack_trace' in node.meta and node.meta['stack_trace']:
            for line in node.meta['stack_trace'].strip().split('\n'):
                if 'File "' in line and ', line ' in line:
                    parts = line.strip().split('"')
                    if len(parts) >= 2:
                        full_path = parts[1]
                        # Extract just the filename (not full path)
                        file_name = full_path.split('/')[-1]
                        line_num = int(line.split(', line ')[1].split(',')[0])
                        location_info['file'] = file_name
                        location_info['line'] = line_num
                        break

        # Extract module hierarchy from nn_module_stack
        if 'nn_module_stack' in node.meta and node.meta['nn_module_stack']:
            modules = sorted(node.meta['nn_module_stack'].items(), key=lambda x: len(x[1][0]))
            module_classes = []
            module_names = []
            for fqn, (_, class_name) in modules:
                clean = class_name.split('.')[-1] if '.' in class_name else class_name
                if clean not in module_classes:
                    module_classes.append(clean)
                    # Clean torch.fx internal naming to get readable instance names
                    instance = _clean_fx_name(fqn) if fqn else clean.lower()
                    module_names.append(instance)
            if module_classes:
                location_info['module_classes'] = module_classes
                location_info['module_instances'] = module_names

        # Build semantic location in format:
        # "ModuleClass[instance]/SubModule[instance]/forward(file.py:line)/"
        if location_info:
            module_classes = location_info.get('module_classes', [])
            module_instances = location_info.get('module_instances', [])

            # Build module hierarchy
            if module_classes:
                path_parts = []
                for i, mod in enumerate(module_classes):
                    instance = module_instances[i] if i < len(module_instances) else mod.lower()
                    path_parts.append(f"{mod}[{instance}]")
                hierarchy = '/'.join(path_parts)
            else:
                hierarchy = ""

            # Add forward method with file/line info
            file_info = f"forward({file_name}:{line_num})"

            # Combine: hierarchy + forward(file:line) + trailing slash
            if hierarchy:
                semantic_loc = f"{hierarchy}/{file_info}/"
            else:
                semantic_loc = f"{file_info}/"

            # Store for dispatch mode to use
            node.meta['tt_semantic_location'] = semantic_loc
            location_mapping[node.name] = semantic_loc

    graph_module._tt_location_mapping = location_mapping


class FXSemanticDispatchMode(TorchDispatchMode):
    """
    Non-invasive approach: Intercept XLA operations and attach FX semantic
    locations using torch-xla's existing _set_xla_custom_op_name_prefix API.

    This uses the CustomOpNameMetaData mechanism from PRs #5715 and #5838.
    """
    def __init__(self, semantic_locations, fx_node_info=None):
        super().__init__()
        self.semantic_locations = semantic_locations
        self.fx_node_info = fx_node_info or []
        self.operation_index = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        debug = os.environ.get("TT_DEBUG_DISPATCH", "1") == "1"

        # Execute the operation normally
        res = func(*args, **kwargs)

        # Skip internal operations that don't correspond to FX nodes
        if 'mark_argument_attributes' in func.__name__:
            if debug:
                print(f"[DISPATCH skip] {func.__name__}", flush=True)
            return res

        # Get semantic location for this operation (before incrementing)
        semantic_loc = None
        if self.operation_index < len(self.semantic_locations):
            semantic_loc = self.semantic_locations[self.operation_index]

        if debug:
            print(f"[DISPATCH {self.operation_index}] {func.__name__}", flush=True)

            # Verify order: check if dispatched op matches expected FX node
            if self.operation_index < len(self.fx_node_info):
                expected = self.fx_node_info[self.operation_index]
                expected_op = expected['op_name']
                actual_op = func.__name__

                # Check if operation names match (allowing for variants like .default, .Tensor, etc.)
                ops_match = (expected_op == actual_op or
                            expected_op in actual_op or
                            actual_op in expected_op)

                if ops_match:
                    print(f"  -> ✓ Order matches: FX[{expected['fx_index']}] {expected['node_name']} ({expected_op})", flush=True)
                else:
                    print(f"  -> ✗ ORDER MISMATCH!", flush=True)
                    print(f"     Expected FX[{expected['fx_index']}]: {expected['node_name']} ({expected_op})", flush=True)
                    print(f"     Got dispatch: {actual_op}", flush=True)

            if semantic_loc:
                # Show which semantic location we're using
                print(f"  -> Using semantic_loc: {semantic_loc}", flush=True)

        # Check if result is a tensor (or tuple of tensors) and set metadata
        def try_set_metadata(tensor):
            """Try to set metadata on a single tensor."""
            if not isinstance(tensor, torch.Tensor):
                return False
            if not hasattr(tensor, 'device'):
                return False
            try:
                # Check if it's an XLA tensor
                if 'xla' in str(tensor.device):
                    if semantic_loc:
                        # Use torch-xla's EXISTING API (non-invasive!)
                        success = torch_xla._XLAC._set_xla_custom_op_name_prefix(
                            tensor, semantic_loc, 0
                        )
                        if debug:
                            print(f"  -> Set metadata on XLA tensor (success={success})", flush=True)
                    return True
                else:
                    if debug:
                        print(f"  -> Skipped: CPU tensor (device={tensor.device})", flush=True)
                    return False
            except Exception as e:
                # If anything goes wrong, just skip this tensor
                if debug:
                    print(f"  -> Error setting metadata: {e}")
                return False
            return False

        # Handle single tensor result
        set_metadata_success = False
        if try_set_metadata(res):
            set_metadata_success = True
        # Handle tuple/list of tensors
        elif isinstance(res, (tuple, list)):
            for item in res:
                if try_set_metadata(item):
                    set_metadata_success = True
                    break  # Only try first XLA tensor

        # ✅ ALWAYS increment counter (for both XLA and CPU operations)
        self.operation_index += 1

        return res


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

        # Extract semantic locations from FX graph (once during initialization)
        # Only include nodes that actually have semantic locations (skip mark_argument_attributes, etc.)
        self.semantic_locations = []
        self.fx_node_info = []  # For verification: store (node_name, op_name)

        for node in module.graph.nodes:
            if node.op == 'call_function' and 'tt_semantic_location' in node.meta:
                self.semantic_locations.append(node.meta['tt_semantic_location'])

                # Store node info for order verification
                op_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
                self.fx_node_info.append({
                    'fx_index': len(self.fx_node_info),
                    'node_name': node.name,
                    'op_name': op_name,
                    'semantic_location': node.meta['tt_semantic_location']
                })

    def __call__(self, *args):
        # Execute FX graph with dispatch mode intercepting operations
        # Only enable semantic location tracking if XLA_HLO_DEBUG is enabled
        if self.semantic_locations and os.environ.get("XLA_HLO_DEBUG", "1") == "1":
            with FXSemanticDispatchMode(self.semantic_locations, self.fx_node_info):
                output = self.module(*args)
        else:
            output = self.module(*args)

        # This tells torch-xla to cut the graph at only what is required to
        # compute all tensors in the `output` list.
        torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
        return output


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options=None):

    module = torch_pass_pipeline(gm, example_inputs)
    return XLAExecutor(module)
