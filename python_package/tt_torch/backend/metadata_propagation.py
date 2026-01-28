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
2. Using a custom FX Interpreter to track which FX node is being executed
3. Intercepting operations at runtime via TorchDispatchMode
4. Attaching node metadata (module hierarchy, file, line) to XLA tensors using torch-xla's API

The Interpreter-based approach ensures correct metadata attribution even when a single
FX node decomposes into multiple aten operations at dispatch time. A context variable
tracks the currently executing FX node, and all dispatched ops inherit its metadata.

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

import ast
import os
import types
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import torch_xla
from torch.fx import Interpreter
from torch.utils._python_dispatch import TorchDispatchMode
from ttxla_tools.logging import logger

# Enable debug logging for location metadata
DBG_LOC = False

# Context variable to track current FX node metadata across dispatch calls.
# This allows MetadataDispatchMode to know which FX node triggered each dispatch,
# even when a single FX node decomposes into multiple aten operations.
_current_node_metadata: ContextVar[str | None] = ContextVar(
    "_current_node_metadata", default=None
)


@dataclass
class EmitModuleLoc:
    module_class: str
    module_name: str


@dataclass
class EmitLoc:
    modules: list[EmitModuleLoc]
    func_path: str
    func_name: str
    op_line_num: int
    op_name: str
    op_index: int = -1

    @staticmethod
    def make_unknown() -> "EmitLoc":
        return EmitLoc(
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

        # # Don't print op name, it gets added later by torch-xla
        # return f"{self.op_index}{SEPARATOR}{modules_str}{self.func_path}{SEPARATOR}{self.func_name}{SEPARATOR}{self.op_line_num}{SEPARATOR}"

        # Actually print the op name as well, to help debug mismatches
        return f"{self.op_index}{SEPARATOR}{modules_str}{self.func_path}{SEPARATOR}{self.func_name}{SEPARATOR}{self.op_line_num}{SEPARATOR}{self.op_name}"

    def __repr__(self) -> str:
        return self.to_string()

    def __str__(self) -> str:
        return self.to_string()


def _find_enclosing_function(
    full_path: str, line_num: int, mode: str = "simple"
) -> tuple[str, str]:
    """
    Given a file path and a line number, returns the full path (with line number) of the enclosing function.
    If not found or file cannot be opened, returns "unknown".

    Args:
        full_path: Path to the source file.
        line_num: The 1-based line number for which to find the enclosing function.
        mode: 'simple' (default) uses a line-by-line scan;
              'ast' uses the Python AST to determine the most accurate enclosing function.

    Returns:
        str: tuple of (full_path:str, func_name:str) or ("unknown", "unknown")
    """
    if mode == "simple":
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

    elif mode == "ast":
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
                        max(getattr(x, "lineno", node.lineno) for x in node.body)
                        if node.body
                        else node.lineno
                    ):
                        # Recursively visit all children to find more specific (inner) functions.
                        # This includes functions nested directly AND methods inside nested classes.
                        self.generic_visit(node)

                        # If no more specific one found, record this one
                        if self.found is None or node.lineno > (self.found_lineno or 0):
                            self.found = node
                            self.found_lineno = node.lineno
                            self.found_name = node.name

                # Also handle async functions
                visit_AsyncFunctionDef = visit_FunctionDef

            visitor = LineFunctionVisitor(line_num)
            visitor.visit(tree)
            if visitor.found is not None:
                return f"{full_path}:{visitor.found_lineno}", visitor.found_name
            else:
                return "unknown", "unknown"
        except Exception:
            return "unknown", "unknown"
    else:
        raise ValueError(
            'Invalid mode for _find_enclosing_function: choose "simple" or "ast"'
        )


def _extract_source_and_module_hierarchy_info(
    node: torch.fx.Node, op_index: int = -1
) -> EmitLoc:
    if "stack_trace" not in node.meta or not node.meta["stack_trace"]:
        return EmitLoc.make_unknown()

    global DBG_LOC

    # Process in reverse to get deepest (innermost) call first
    lines = node.meta["stack_trace"].strip().split("\n")

    # Search the stack trace to find the reference to line-of-code where the op is called.
    # Unfortunately, this is not straightforward to do, as stack trace can be "muddled" by other things, like torch overrides, etc.
    #
    # Initially, the logic was to take the last line of the stack trace that started with "File ", which worked for some cases, but not all.
    #
    # Since it is not obvious how to extract a valid trace, we will keep a record of types of stack traces encountered so far, and write our heuristic so that it works for all of them.
    #
    # ====
    #
    # Case 1:
    # node.meta["stack_trace"].strip().split("\n")
    # 0 = 'File "/localdev/svuckovic/_workspace/repos/tt-xla/examples/pytorch/codegen/recover_structure.py", line 47, in forward'
    # 1 = '    x = self.m1(x)'
    # 2 = '  File "/localdev/svuckovic/_workspace/repos/tt-xla/examples/pytorch/codegen/recover_structure.py", line 28, in forward'
    # 3 = '    return x * self.w'
    # 4 = '  File "/localdev/svuckovic/_workspace/repos/tt-xla/python_package/tt_torch/torch_overrides.py", line 22, in __torch_function__'
    # 5 = '    return func(*args, **(kwargs or {}))'
    #
    # In the above case, the valid trace is the last line that starts with "File " but isn't in the torch_overrides.py file.
    #
    # ====
    #
    # Case 2:
    # node.meta["stack_trace"].strip().split("\n")
    # 0 = 'File "/localdev/svuckovic/_workspace/repos/tt-xla/examples/pytorch/codegen/python/custom_module.py", line 33, in forward'
    # 1 = '    return torch.sum(x**2)'
    # 2 = '  File "/localdev/svuckovic/_workspace/repos/tt-xla/python_package/tt_torch/torch_overrides.py", line 22, in __torch_function__'
    # 3 = '    return func(*args, **(kwargs or {}))'
    # 4 = '  File "/localdev/svuckovic/_workspace/repos/tt-xla/venv/lib/python3.11/site-packages/torch/_tensor.py", line 39, in wrapped'
    # 5 = '    return f(*args, **kwargs)'
    #
    # In the above case, the valid trace is the last line that starts with "File " but isn't in the torch_overrides.py file.
    #
    # ====
    #
    # Case 3:
    # 0 = 'File "/localdev/svuckovic/_workspace/repos/tt-xla/examples/pytorch/codegen/python/custom_module.py", line 33, in forward'
    # 1 = '    return torch.sum(x**2)'
    # 2 = '  File "/localdev/svuckovic/_workspace/repos/tt-xla/python_package/tt_torch/torch_overrides.py", line 22, in __torch_function__'
    # 3 = '    return func(*args, **(kwargs or {}))'
    # 4 = '  File "/usr/local/lib/python3.11/dist-packages/torch/_tensor.py", line 39, in wrapped'
    # 5 = '    return f(*args, **kwargs)'
    #
    # ====
    #
    # Files to skip when searching for the source location.
    # These are internal files that appear in stack traces but don't represent user code.
    skip_patterns = [
        "python_package/tt_torch/torch_overrides.py",  # Case 1: our custom torch overrides (e.g., tt-xla/python_package/tt_torch/torch_overrides.py)
        "site-packages/torch/",  # Case 2: Internal torch files (e.g., venv/.../torch/_tensor.py)
        "dist-packages/torch/",  # Case 3: Internal torch files (e.g., /usr/local/lib/.../torch/_tensor.py)
        "-packages/transformers/activations.py",  # this fixes gelu, find a better pattern to describe this (all of transformers, but not models?)
        "-packages/transformers/integrations/",  # this fixes sdpa op
    ]
    line = next(
        (
            line
            for line in reversed(lines)
            if (stripped := line.strip()).startswith('File "')
            and len(stripped.split(",")) >= 3
            and not any(skip_pattern in stripped for skip_pattern in skip_patterns)
        ),
        None,
    )

    if line is None:
        return EmitLoc.make_unknown()

    DBG_LOC and print(f"Printing stack trace line: {line}")
    stripped = line.strip()
    parts = stripped.split(",")

    # Parse the line to get the full path, line number, and function name
    # File "/path/file.py", line 42, in forward
    full_path = parts[0].split('"')[1]
    line_num = int(parts[1].split()[-1])
    func_name = parts[2].split()[-1]
    func_path, found_func_name = _find_enclosing_function(full_path, line_num)
    func_path_ast, found_func_name_ast = _find_enclosing_function(
        full_path, line_num, mode="ast"
    )

    # If either of the paths is "unknown", return unknown loc
    if "unknown" in (func_path_ast, found_func_name_ast):
        DBG_LOC and print(
            f"  unknown function name or path: {func_name}, {found_func_name}, {func_path_ast}, {found_func_name_ast}"
        )
        logger.debug(
            f"Could not find file path or function name for node - location info will be unknown - this will affect codegen"
        )
        return EmitLoc.make_unknown()

    # If the function names don't match, raise an error
    if func_name != found_func_name:
        DBG_LOC and print(f"  function name mismatch: {func_name}, {found_func_name}")
        logger.debug(
            "Function name mismatch between stack_trace and found_func_name modes\n"
            f"stack_trace: {func_name}\n"
            f"found_func_name: {found_func_name}\n"
            f"found_func_name_ast: {found_func_name_ast}"
        )
        return EmitLoc.make_unknown()

    # If the function paths don't match, raise an error
    if func_path != func_path_ast:
        DBG_LOC and print(f"  func_path: {func_path}, {found_func_name}")
        DBG_LOC and print(f"  func_path_ast: {func_path_ast}, {found_func_name_ast}")
        raise ValueError(
            f"Function path mismatch for {full_path}:{line_num} between simple and ast modes\nSimple: {func_path}\n   Ast: {func_path_ast}"
        )

    # Extract module hierarchy from node's nn_module_stack metadata.
    extracted_modules = []
    if "nn_module_stack" in node.meta and node.meta["nn_module_stack"]:
        # Sort by path length to get hierarchy order (shorter = outer, longer = inner modules)
        # Format: (path, class_name) e.g., ("L['self'].inner.linear", "torch.nn.modules.linear.Linear")
        modules = sorted(node.meta["nn_module_stack"].values(), key=lambda x: len(x[0]))

        DBG_LOC and print(f"  Printing modules:")
        for path, class_name in modules:
            DBG_LOC and print(f"    path: {path}")
            DBG_LOC and print(f"    class_name: {class_name}")
            module_class = (
                class_name.split(".")[-1] if "." in class_name else class_name
            )

            # Extract instance from path (e.g., "L['self'].inner.linear" → "inner.linear")
            module_name = (
                path.replace("L['self'].", "").replace("L['self']", "") if path else ""
            )
            module_name = module_name if module_name else module_class.lower()

            extracted_modules.append(EmitModuleLoc(module_class, module_name))

    return EmitLoc(
        modules=extracted_modules,
        func_path=func_path,
        func_name=func_name,
        op_line_num=line_num,
        op_name=node.name,
        op_index=op_index,
    )


def extract_nodes_info(graph_module: torch.fx.GraphModule) -> dict[str, str]:
    """
    Extract metadata for each FX node, returning a dict keyed by node name.

    Returns a dict mapping node name -> metadata string, which allows the
    MetadataInterpreter to look up metadata by node name during execution.
    This approach correctly handles FX nodes that decompose into multiple
    aten operations at dispatch time.
    """
    global DBG_LOC

    node_info: dict[str, str] = {}
    op_index = 0

    for node in graph_module.graph.nodes:

        DBG_LOC and print(f"node.op: {node.op}")
        DBG_LOC and print(f"  node.name: {node.name}")

        # Only process call_function nodes
        if node.op != "call_function":
            DBG_LOC and print(f"  SKIPPING node: {node.op} - {node.name}")
            continue

        # If no metadata is available, use unknown location
        if not hasattr(node, "meta") or not node.meta:
            DBG_LOC and print(f"  NO meta for node: {node.op} - {node.name}")
            node_info[node.name] = EmitLoc.make_unknown().to_string()
            op_index += 1
            continue

        # Skip if node.target is <class 'builtin_function_or_method'>
        if isinstance(node.target, types.BuiltinFunctionType):
            DBG_LOC and print(
                f"  SKIPPING node: {node.op} - {node.name} - it is a builtin function"
            )
            continue
        DBG_LOC and print(f"  node.target: {node.target}")

        # Extract metadata components
        emit_loc = _extract_source_and_module_hierarchy_info(node, op_index)

        DBG_LOC and print(f"  EmitLoc: {emit_loc}")

        # Build location string if we have any metadata
        node_info[node.name] = emit_loc.to_string()
        op_index += 1

    return node_info


class MetadataInterpreter(Interpreter):
    """
    Executes FX graph while setting metadata context for each node.

    This ensures that all dispatched aten operations from a single FX node
    receive the same metadata, regardless of how many ops the FX node
    decomposes into at runtime.

    For example, torch.matmul may decompose into expand -> view -> bmm -> view
    at dispatch time. Using this Interpreter, all four aten ops will receive
    the metadata from the original matmul FX node.
    """

    def __init__(self, module: torch.fx.GraphModule, node_info: dict[str, str]):
        super().__init__(module)
        # Map from node name -> metadata string
        self.node_info = node_info

    def run_node(self, node: torch.fx.Node):
        if node.op == "call_function":
            # Get metadata for this node
            metadata = self.node_info.get(node.name)

            if metadata:
                # Set context before executing - all dispatched ops will see this
                token = _current_node_metadata.set(metadata)
                try:
                    return super().run_node(node)
                finally:
                    _current_node_metadata.reset(token)

        return super().run_node(node)


class MetadataDispatchMode(TorchDispatchMode):
    """
    Intercept XLA operations at runtime and attach FX node metadata to operations in HLO IR.

    TorchDispatchMode hooks into PyTorch's dispatcher: when active, every tensor operation
    triggers __torch_dispatch__ before executing. We use this to inject metadata without
    modifying the computation itself.

    How it works:
    1. At compile time, extract_nodes_info() builds a dict mapping node names to metadata
       strings from FX graph nodes.

    2. At runtime, MetadataInterpreter executes the graph and sets a context variable
       (_current_node_metadata) before each FX node executes. This context variable
       tracks which FX node is currently being executed.

    3. When aten operations are dispatched (potentially multiple ops from a single FX node
       due to decomposition), this class reads the context variable to get the correct
       metadata for the currently executing FX node.

    4. For each dispatched operation, we attach metadata to the output XLA tensor using
       torch_xla._XLAC._set_xla_custom_op_name_prefix(). torch-xla traces back from the
       tensor to find which operation produced it, then labels that operation node in HLO.

    This approach correctly handles FX nodes that decompose into multiple aten operations,
    unlike a counter-based approach which assumes 1:1 mapping between FX nodes and dispatches.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)

        # Get metadata from context variable (set by MetadataInterpreter)
        metadata = _current_node_metadata.get()
        if metadata:
            self._set_metadata(result, metadata)

        return result

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
        if isinstance(result, torch.Tensor):
            self._set_tensor_metadata(result, module_hierarchy)
        elif isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, torch.Tensor):
                    if self._set_tensor_metadata(item, module_hierarchy):
                        break  # One output labels the entire operation node

    def _set_tensor_metadata(self, tensor: torch.Tensor, module_hierarchy: str) -> bool:
        try:
            if "xla" in str(tensor.device):
                torch_xla._XLAC._set_xla_custom_op_name_prefix(
                    tensor, module_hierarchy, 0
                )
                return True
        except Exception:
            logger.error(f"Error setting metadata - ({module_hierarchy})")

        return False
