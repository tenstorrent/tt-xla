# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
FX node metadata tracking for torch-xla operations.

This module provides node metadata tracking by:
1. Extracting location metadata from FX graph nodes (module hierarchy, file, line)
2. Intercepting operations at runtime via TorchDispatchMode
3. Attaching node metadata (module hierarchy, file, line) to XLA tensors using torch-xla's API

The counter-based approach relies on FX guarantees:
- Nodes are stored in topological order
- Code generation preserves node order
- Dispatch happens in execution order

Note: extract_nodes_info() must be called after all FX transformation passes complete.
FX passes may modify the graph (add/remove/reorder nodes), so extracting metadata too
early would create misalignment with runtime execution order.

Correct usage pattern:
    graph = apply_pass_1(graph)
    graph = apply_pass_2(graph)
    ...
    graph.recompile()              # Finalize the graph
    nodes_info = extract_nodes_info(graph)  # Extract after finalization

"""
import torch
import torch_xla
from torch.utils._python_dispatch import TorchDispatchMode


def _extract_source_info(node: torch.fx.Node) -> tuple[str, int, str]:
    """
    Extract source file, line number, and function name from node's stack trace.
    Processes the stack trace in reverse to find the DEEPEST call (innermost module).
    Stack trace line format: '  File "/path/to/file.py", line 42, in forward'

    Returns:
        Tuple of (file_name, line_num, func_name)
    """
    if 'stack_trace' not in node.meta or not node.meta['stack_trace']:
        return "unknown", 0, "unknown"

    # Process in reverse to get deepest (innermost) call first
    lines = node.meta['stack_trace'].strip().split('\n')
    for line in reversed(lines):
        stripped = line.strip()

        if not stripped.startswith('File "'):
            continue

        parts = stripped.split(',')
        if len(parts) < 3:
            continue

        # Parse: File "/path/file.py", line 42, in forward
        full_path = parts[0].split('"')[1]
        file_name = full_path.split('/')[-1]
        line_num = int(parts[1].split()[-1])
        func_name = parts[2].split()[-1]

        return file_name, line_num, func_name

    return "unknown", 0, "unknown"


def _extract_module_hierarchy(node: torch.fx.Node) -> tuple[list[str], list[str]]:
    """
    Extract module hierarchy from node's nn_module_stack metadata.

    Returns:
        Tuple of (module_classes, module_names)
    """
    module_classes: list[str] = []
    module_names: list[str] = []

    if 'nn_module_stack' not in node.meta or not node.meta['nn_module_stack']:
        return module_classes, module_names

    # Sort by path length to get hierarchy order (shorter = outer, longer = inner modules)
    # Format: (path, class_name) e.g., ("L['self'].inner.linear", "torch.nn.modules.linear.Linear")
    modules = sorted(node.meta['nn_module_stack'].values(), key=lambda x: len(x[0]))

    for path, class_name in modules:
        module_class = class_name.split('.')[-1] if '.' in class_name else class_name
        module_classes.append(module_class)

        # Extract instance from path (e.g., "L['self'].inner.linear" → "inner.linear")
        instance = path.replace("L['self'].", "").replace("L['self']", "") if path else ""
        instance = instance if instance else module_class.lower()
        module_names.append(instance)

    return module_classes, module_names


def _build_location_string(
    module_classes: list[str],
    module_names: list[str],
    file_name: str,
    line_num: int,
    func_name: str
) -> str:
    """
    Build module hierarchy string from components.

    Format: "ModuleClass[mod_name_1]/SubModule[mod_name_2]/func_name(file.py:line)/"
    """
    path_parts = []

    if module_classes:
        for mod_class, mod_name in zip(module_classes, module_names):
            path_parts.append(f"{mod_class}[{mod_name}]")

    hierarchy = '/'.join(path_parts) if path_parts else ""
    file_info = f"{func_name}({file_name}:{line_num})"

    if hierarchy:
        return f"{hierarchy}/{file_info}/"
    else:
        return f"{file_info}/"


def extract_nodes_info(graph_module: torch.fx.GraphModule) -> list[str]:
    """
    Extract node metadata from FX graph nodes.

    Returns ordered list of module hierarchy for call_function nodes only.
    Filtering matches dispatch behavior: only call_function nodes dispatch operations.

    Format: "ModuleClass[instance]/SubModule[instance]/func_name(file.py:line)/"

    This prefix is concatenated with operation type by torch-xla:
    "Linear[linear]/forward(model.py:42)/" + "aten__mm"
    → "Linear[linear]/forward(model.py:42)/aten__mm"
    """
    nodes_info = []

    for node in graph_module.graph.nodes:
        # Only process call_function nodes
        if node.op != 'call_function':
            continue

        if not hasattr(node, 'meta') or not node.meta:
            continue

        # Extract metadata components
        file_name, line_num, func_name = _extract_source_info(node)
        module_classes, module_names = _extract_module_hierarchy(node)

        # Build location string if we have any metadata
        if file_name != "unknown" or module_classes:
            location = _build_location_string(
                module_classes, module_names, file_name, line_num, func_name
            )
            nodes_info.append(location)

    return nodes_info


class MetadataDispatchMode(TorchDispatchMode):
    """
    Intercept XLA operations at runtime and attach FX node metadata to operations in HLO IR.

    TorchDispatchMode hooks into PyTorch's dispatcher: when active, every tensor operation
    triggers __torch_dispatch__ before executing. We use this to inject metadata without
    modifying the computation itself.

    How it works:
    1. At compile time, extract_nodes_info() builds an ordered list of metadata from FX graph
       nodes (only call_function nodes, since they're the only ones that dispatch operations).

    2. At runtime, when graph_module.forward() executes, it triggers operations in sequence.
       Each operation increments our counter, letting us map it to the corresponding FX node.

       Example flow:
         FX Node 0 (call_function: linear) → Runtime: aten::linear dispatched   (counter=0)
         FX Node 1 (call_function: relu)   → Runtime: aten::relu dispatched     (counter=1)
         FX Node 2 (output)                 → No dispatch (not call_function)

       The counter-based approach works because FX guarantees topological ordering and
       execution follows the same order.

    3. For each dispatched operation, we attach metadata to the output XLA tensor using
       torch_xla._XLAC._set_xla_custom_op_name_prefix(). torch-xla traces back from the
       tensor to find which operation produced it, then labels that operation node in HLO.

       Result in HLO: "Linear[fc]/forward(model.py:12)/aten__linear"
                      "ReLU[act]/forward(model.py:13)/aten__relu"

    This is non-invasive: it doesn't modify the FX graph or operation semantics, only adds
    debugging metadata to the generated XLA HLO IR for better observability.
    """

    def __init__(self, node_info: list[str]):
        super().__init__()
        self.node_info = node_info
        self.operation_index = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        res = func(*args, **kwargs)

        # Skip tt operations that are just markers for XLA tensors
        if 'tt' in func.__name__:
            return res

        # Get semantic location for this operation
        module_hierarchy = None
        if self.operation_index < len(self.node_info):
            module_hierarchy = self.node_info[self.operation_index]

        # Set metadata on XLA tensors
        if module_hierarchy:
            self._set_metadata(res, module_hierarchy)

        # increment counter (critical for correctness)
        self.operation_index += 1

        return res

    def _set_metadata(self, result: torch.Tensor | tuple | list, module_hierarchy: str) -> None:
        """
        Set semantic location metadata on XLA tensors in result.

        For operations returning multiple tensors (e.g., torch.sort -> (values, indices)),
        only the first XLA tensor needs metadata. torch-xla traces back from the output
        tensor to find the operation node that produced it, then labels that operation node
        in the HLO IR. Since all outputs come from the same operation node, setting metadata
        on one output is sufficient to label the entire operation.
        """
        if isinstance(result, torch.Tensor):
            self._set_tensor_metadata(result, module_hierarchy)
        elif isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, torch.Tensor):
                    if self._set_tensor_metadata(item, module_hierarchy):
                        break  # One output labels the entire operation node

    def _set_tensor_metadata(self, tensor: torch.Tensor, module_hierarchy: str) -> bool:
        try:
            if 'xla' in str(tensor.device):
                torch_xla._XLAC._set_xla_custom_op_name_prefix(tensor, module_hierarchy, 0)
                return True
        except Exception:
            print(f"Error setting metadata - ({module_hierarchy})", flush=True)

        return False
