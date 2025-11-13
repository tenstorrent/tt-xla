# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
FX node metadata tracking for torch-xla operations.

NOTE: This module is only used for debugging purposes as it affects performance.
In order to use it you have to set the environment variable XLA_HLO_DEBUG to 1 by running the following command:
export XLA_HLO_DEBUG=1

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
import types
from torch.utils._python_dispatch import TorchDispatchMode
from dataclasses import dataclass

UNKNOWN_LOCATION = "unknown/unknown(unknown:0)/"


import ast

def _find_enclosing_function(full_path: str, line_num: int, mode: str = 'simple') -> str:
    """
    Given a file path and a line number, returns the full path (with line number) of the enclosing function.
    If not found or file cannot be opened, returns "unknown".

    Args:
        full_path: Path to the source file.
        line_num: The 1-based line number for which to find the enclosing function.
        mode: 'simple' (default) uses a line-by-line scan;
              'ast' uses the Python AST to determine the most accurate enclosing function.

    Returns:
        str: "<full_path>:<func_line>/<func_name>" or "unknown"
    """
    if mode == 'simple':
        try:
            with open(full_path, "r") as f:
                current_func_name = "unknown"
                current_func_lineno = 0
                for lineno, file_line in enumerate(f, 1):
                    stripped = file_line.lstrip()
                    if stripped.startswith("def ") and "(" in stripped[4:]:
                        func_def = stripped[4:].split("(")[0].strip()
                        current_func_name = func_def
                        current_func_lineno = lineno
                    if lineno == line_num:
                        break
                if current_func_name != "unknown":
                    return f"{full_path}:{current_func_lineno}", current_func_name
                else:
                    return "unknown", "unknown"
        except Exception:
            return "unknown", "unknown"

    elif mode == 'ast':
        try:
            with open(full_path, "r") as f:
                source = f.read()
            tree = ast.parse(source, full_path)

            class LineFunctionVisitor(ast.NodeVisitor):
                def __init__(self, line):
                    self.line = line
                    self.found = None
                    self.found_lineno = None
                    self.found_name = None

                def visit_FunctionDef(self, node):
                    if hasattr(node, "body") and node.lineno <= self.line <= (
                        max(getattr(x, "lineno", node.lineno) for x in node.body) if node.body else node.lineno
                    ):
                        # Possibly nested, visit children to check for more specific (inner) function
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child, ast.FunctionDef):
                                self.visit(child)
                        # If no more specific one found, record this one
                        if self.found is None or node.lineno > (self.found_lineno or 0):
                            self.found = node
                            self.found_lineno = node.lineno
                            self.found_name = node.name

            visitor = LineFunctionVisitor(line_num)
            visitor.visit(tree)
            if visitor.found is not None:
                return f"{full_path}:{visitor.found_lineno}", visitor.found_name
            else:
                return "unknown", "unknown"
        except Exception:
            return "unknown", "unknown"
    else:
        raise ValueError('Invalid mode for _find_enclosing_function: choose "simple" or "ast"')

@dataclass
class LocationModuleCodegen:
    module_class: str
    module_name: str

@dataclass
class LocationCodegen:
    modules: list[LocationModuleCodegen]
    func_path: str
    func_name: str
    op_line_num: int
    op_name: str

    def make_unknown(self) -> 'LocationCodegen':
        return LocationCodegen(
            modules=[],
            func_path="unknown",
            func_name="unknown",
            op_line_num=-1,
            op_name="unknown",
        )

    def to_string(self) -> str:
        SEPARATOR = "|"
        modules_list = []
        for mod in self.modules:
            modules_list.append(f"{mod.module_class}[{mod.module_name}]")
        # Add separator to the end of the modules_str
        modules_str = SEPARATOR.join(modules_list) + SEPARATOR if modules_list else ""
        return f"{modules_str}{self.func_path}{SEPARATOR}{self.func_name}{SEPARATOR}{self.op_line_num}{SEPARATOR}{self.op_name}"

    def __repr__(self) -> str:
        return self.to_string()

    def __str__(self) -> str:
        return self.to_string()


def _extract_source_info(node: torch.fx.Node) -> LocationCodegen:
    """
    Extract source file, line number, and function name from node's stack trace.
    Processes the stack trace in reverse to find the DEEPEST call (innermost module).
    Stack trace line format: '  File "/path/to/file.py", line 42, in forward'

    Returns:
        LocationCodegen
    """
    if "stack_trace" not in node.meta or not node.meta["stack_trace"]:
        return LocationCodegen.make_unknown()

    # Process in reverse to get deepest (innermost) call first
    lines = node.meta["stack_trace"].strip().split("\n")
    print(f"Printing lines:")
    for line in reversed(lines):
        print(f"  line: {line}")
        stripped = line.strip()

        if not stripped.startswith('File "'):
            continue

        parts = stripped.split(",")
        if len(parts) < 3:
            continue

        # Parse: File "/path/file.py", line 42, in forward
        full_path = parts[0].split('"')[1]
        file_name = full_path.split("/")[-1]
        line_num = int(parts[1].split()[-1])
        func_name = parts[2].split()[-1]
        func_path, func_name = _find_enclosing_function(full_path, line_num)
        func_path_ast, func_name_ast = _find_enclosing_function(full_path, line_num, mode='ast')
        if func_path != func_path_ast:
            print(f"  func_path: {func_path}, {func_name}")
            print(f"  func_path_ast: {func_path_ast}, {func_name_ast}")
            raise ValueError(f"Function path mismatch for {full_path}:{line_num} between simple and ast modes")

        location_codegen = LocationCodegen(
            modules=[],
            func_path=func_path,
            func_name=func_name,
            op_line_num=line_num,
            op_name="",
        )

        # return file_name, line_num, func_name
        # return full_path, line_num, func_path, func_name
        return location_codegen

    return LocationCodegen.make_unknown()


def _extract_module_hierarchy(node: torch.fx.Node, location_codegen: LocationCodegen) -> LocationCodegen:
    """
    Extract module hierarchy from node's nn_module_stack metadata.

    Returns:
        LocationCodegen
    """
    module_classes: list[str] = []
    module_names: list[str] = []

    if "nn_module_stack" not in node.meta or not node.meta["nn_module_stack"]:
        return location_codegen

    # Sort by path length to get hierarchy order (shorter = outer, longer = inner modules)
    # Format: (path, class_name) e.g., ("L['self'].inner.linear", "torch.nn.modules.linear.Linear")
    modules = sorted(node.meta["nn_module_stack"].values(), key=lambda x: len(x[0]))

    print(f"    node.name: {node.name}")
    print(f"    Printing modules:")
    for path, class_name in modules:
        print(f"      path: {path}")
        print(f"      class_name: {class_name}")
        module_class = class_name.split(".")[-1] if "." in class_name else class_name

        # Extract instance from path (e.g., "L['self'].inner.linear" → "inner.linear")
        module_name = (
            path.replace("L['self'].", "").replace("L['self']", "") if path else ""
        )
        module_name = module_name if module_name else module_class.lower()
        location_codegen.modules.append(LocationModuleCodegen(module_class, module_name))

    return location_codegen


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

        print(f"  node.op: {node.op}")
        # print(f"  node.target: {node.target}")
        # print(f"  node.args: {node.args}")
        # print(f"  node.kwargs: {node.kwargs}")
        # print(f"  node.meta: {node.meta}")
        print(f"  node.name: {node.name}")
        # print(f"  node.type: {node.type}")

        # Only process call_function nodes
        if node.op != "call_function":
            print(f"  SKIPPING node.name: {node.name}")
            continue

        if not hasattr(node, "meta") or not node.meta:
            print(f"  NO meta for node.name: {node.name}")
            nodes_info.append(UNKNOWN_LOCATION)
            continue

        # Check if node.target is <class 'builtin_function_or_method'>
        if isinstance(node.target, types.BuiltinFunctionType):
            print(f"  SKIPPING node.name: {node.name} because it is a builtin function")
            nodes_info.append(UNKNOWN_LOCATION)
            continue
        print(f"  node.target: {node.target}")

        # Extract metadata components
        location_codegen = _extract_source_info(node)
        location_codegen = _extract_module_hierarchy(node, location_codegen)

        # if location_codegen.op_name == "aten__convolution_overrideable_input":
        print(f"  HERE!!")
        print(f"  location_codegen: {location_codegen}")

        # Build location string if we have any metadata
        nodes_info.append(location_codegen.to_string())

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

    def __torch_dispatch__(self, func, types, args=(), kwargs={}):
        res = func(*args, **kwargs)

        # print(f"  func: {func}")

        # Get semantic location for this operation
        module_hierarchy = UNKNOWN_LOCATION
        if self.operation_index < len(self.node_info):
            module_hierarchy = self.node_info[self.operation_index]

        # Set metadata only for computation nodes since they have module hierarchy info.
        # XLA nodes without location info inherit from parent/child nodes, which works
        # perfectly since computation nodes always have locations.
        if module_hierarchy != UNKNOWN_LOCATION:
            # print(f"  SETTING metadata for node.name: {func}")
            # print(f"  module_hierarchy: {module_hierarchy}")
            self._set_metadata(res, module_hierarchy)

        # increment counter (critical for correctness)
        self.operation_index += 1

        return res

    def _set_metadata(
        self, result: torch.Tensor | tuple | list, module_hierarchy: str
    ) -> None:
        """
        Set semantic location metadata on XLA tensors in result.

        For operations returning multiple tensors (e.g., torch.sort -> (values, indices)),
        only the first XLA tensor needs metadata. torch-xla traces back from the output
        tensor to find the operation node that produced it, then labels that operation node
        in the HLO IR. Since all outputs come from the same operation node, setting metadata
        on one output is sufficient to label the entire operation.
        """
        pass
        if isinstance(result, torch.Tensor):
            self._set_tensor_metadata(result, module_hierarchy)
            # pass
        elif isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, torch.Tensor):
                    if self._set_tensor_metadata(item, module_hierarchy):
                        break  # One output labels the entire operation node

    def _set_tensor_metadata(self, tensor: torch.Tensor, module_hierarchy: str) -> bool:
        # assert False, "Not implemented"
        try:
            if "xla" in str(tensor.device):
                torch_xla._XLAC._set_xla_custom_op_name_prefix(
                    tensor, module_hierarchy, 0
                )
                return True
        except Exception:
            print(f"Error setting metadata - ({module_hierarchy})", flush=True)

        return False
