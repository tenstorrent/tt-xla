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
import ast
import logging
import os
import types
from dataclasses import dataclass

import torch
import torch_xla
from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger(__name__)

# Enable debug logging for location metadata
DBG_LOC = False


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

        # Don't print op name, it gets added later by torch-xla
        return f"{self.op_index}{SEPARATOR}{modules_str}{self.func_path}{SEPARATOR}{self.func_name}{SEPARATOR}{self.op_line_num}{SEPARATOR}"

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
    # 0 = 'File "/localdev/svuckovic/_workspace/repos/tt-xla/examples/pytorch/codegen/custom_module.py", line 33, in forward'
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
    # 0 = 'File "/localdev/svuckovic/_workspace/repos/tt-xla/examples/pytorch/codegen/custom_module.py", line 33, in forward'
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
        "dist-packages/transformers/integrations/",
        "dist-packages/transformers/activations.py",
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


@dataclass
class NodeInfo:
    """Holds both the metadata string and the expected op target for validation."""

    metadata: str
    expected_target: str  # e.g., "aten.mm.default" or "aten.add.Tensor"
    node_name: str  # FX node name for debugging


def _normalize_op_name(op_str: str) -> str:
    """
    Normalize op names to allow comparison between FX graph and runtime dispatch.

    FX stores OpOverloadPacket (e.g., "aten.mm") while dispatch uses specific overloads
    (e.g., "aten.mm.default"). We normalize by removing the ".default" suffix and
    handling other common variants.
    """
    # Remove ".default" suffix - this is the most common case
    if op_str.endswith(".default"):
        op_str = op_str[:-8]  # len(".default") == 8
    # Remove ".Tensor" suffix (e.g., "aten.add.Tensor" -> "aten.add")
    elif op_str.endswith(".Tensor"):
        op_str = op_str[:-7]  # len(".Tensor") == 7
    # Remove ".Scalar" suffix
    elif op_str.endswith(".Scalar"):
        op_str = op_str[:-7]

    # XLA converts view ops to copy variants (e.g., aten.view -> aten.view_copy)
    # These are semantically equivalent for our purposes
    copy_variants = [
        ("aten.view_copy", "aten.view"),
        ("aten.permute_copy", "aten.permute"),
        ("aten.expand_copy", "aten.expand"),
        ("aten.reshape_copy", "aten.reshape"),
        ("aten.slice_copy", "aten.slice"),
        ("aten.transpose_copy", "aten.transpose"),
        ("aten.squeeze_copy", "aten.squeeze"),
        ("aten.unsqueeze_copy", "aten.unsqueeze"),
    ]
    for copy_name, orig_name in copy_variants:
        if op_str == copy_name:
            return orig_name

    return op_str


@dataclass
class OpBasedNodeInfo:
    """
    Holds metadata organized by operation name for op-based matching.

    Instead of relying on execution order (counter-based), this allows matching
    by operation name + occurrence index within that operation type.
    """

    # Map from normalized op name -> list of metadata strings (in order of appearance)
    metadata_by_op: dict[str, list[str]]
    # For debugging/validation: total number of FX nodes processed
    total_fx_nodes: int


# Map from decomposed ops to their parent ops for metadata inheritance.
# When XLA decomposes an op at runtime (e.g., matmul -> view + mm + view),
# the decomposed ops can inherit metadata from the parent op.
#
# Note: aten.view is tricky because it exists both in the FX graph AND in matmul
# decomposition. We handle this specially by tracking decomposition state.
DECOMPOSITION_PARENT_MAP = {
    # matmul decomposes to: view -> mm -> _unsafe_view
    # We only map mm and _unsafe_view here; view is handled via state tracking
    "aten.mm": "aten.matmul",
    "aten._unsafe_view": "aten.matmul",
    # bmm may also come from matmul
    "aten.bmm": "aten.matmul",
}

# Ops that signal the START of a matmul decomposition sequence
# When we see a view that overflows AND the next op is mm, we're in matmul decomposition
MATMUL_DECOMP_MIDDLE_OPS = {"aten.mm", "aten.bmm"}
MATMUL_DECOMP_END_OP = "aten._unsafe_view"


def extract_nodes_info(graph_module: torch.fx.GraphModule) -> OpBasedNodeInfo:
    """
    Extract metadata from FX graph nodes, organized by operation name.

    Returns an OpBasedNodeInfo that maps each operation type to a list of
    metadata strings, allowing runtime dispatch to match by op name rather
    than relying on fragile counter-based ordering.
    """
    global DBG_LOC

    metadata_by_op: dict[str, list[str]] = {}
    op_index = 0

    for node in graph_module.graph.nodes:

        DBG_LOC and print(f"node.op: {node.op}")
        DBG_LOC and print(f"  node.name: {node.name}")

        # Only process call_function nodes
        if node.op != "call_function":
            DBG_LOC and print(f"  SKIPPING node: {node.op} - {node.name}")
            continue

        # Skip if node.target is <class 'builtin_function_or_method'>
        if isinstance(node.target, types.BuiltinFunctionType):
            DBG_LOC and print(
                f"  SKIPPING node: {node.op} - {node.name} - it is a builtin function"
            )
            continue

        DBG_LOC and print(f"  node.target: {node.target}")

        # Normalize the operation name for consistent matching
        normalized_op = _normalize_op_name(str(node.target))

        # Extract metadata (or use unknown if not available)
        if not hasattr(node, "meta") or not node.meta:
            DBG_LOC and print(f"  NO meta for node: {node.op} - {node.name}")
            metadata = EmitLoc.make_unknown().to_string()
        else:
            emit_loc = _extract_source_and_module_hierarchy_info(node, op_index)
            DBG_LOC and print(f"  EmitLoc: {emit_loc}")
            metadata = emit_loc.to_string()

        # Add to the op-based map
        if normalized_op not in metadata_by_op:
            metadata_by_op[normalized_op] = []
        metadata_by_op[normalized_op].append(metadata)

        op_index += 1

    return OpBasedNodeInfo(
        metadata_by_op=metadata_by_op,
        total_fx_nodes=op_index,
    )


class MetadataDispatchMode(TorchDispatchMode):
    """
    Intercept XLA operations at runtime and attach FX node metadata to operations in HLO IR.

    TorchDispatchMode hooks into PyTorch's dispatcher: when active, every tensor operation
    triggers __torch_dispatch__ before executing. We use this to inject metadata without
    modifying the computation itself.

    How it works:
    1. At compile time, extract_nodes_info() builds a map from operation names to metadata
       lists. Each op type (e.g., "aten.mm") has an ordered list of metadata for each
       occurrence in the FX graph.

    2. At runtime, when graph_module.forward() executes, operations are dispatched.
       For each operation, we:
       - Normalize the op name (handle .default suffix, view_copy variants, etc.)
       - Look up metadata by (op_name, occurrence_index) where occurrence_index tracks
         how many times we've seen this specific op type

    3. For each dispatched operation, we attach metadata to the output XLA tensor using
       torch_xla._XLAC._set_xla_custom_op_name_prefix(). torch-xla traces back from the
       tensor to find which operation produced it, then labels that operation node in HLO.

    This op-based approach is more robust than counter-based because:
    - It handles XLA runtime decompositions (e.g., matmul -> view + mm + view)
    - Operations that decompose don't throw off the alignment of other ops
    - Each op type maintains its own counter, so decomposed ops just miss metadata
      rather than corrupting all subsequent metadata
    """

    def __init__(self, node_info: OpBasedNodeInfo, validate_alignment: bool = False):
        super().__init__()
        self.node_info = node_info
        # Per-op counters: track how many times each op type has been dispatched
        self.op_counters: dict[str, int] = {}
        # Separate counters for parent op inheritance (e.g., track matmul index for mm ops)
        self.parent_op_counters: dict[str, int] = {}
        # Total operations dispatched (for reporting)
        self.total_dispatched = 0
        # Track ops that had no metadata (for debugging)
        self.ops_without_metadata: list[str] = []
        self.validate_alignment = validate_alignment
        # Track matmul decomposition state: when we see a view overflow followed by mm,
        # we're in a matmul decomposition and should inherit metadata
        self._in_matmul_decomp = False
        self._matmul_decomp_metadata: str | None = None

    def _get_matmul_metadata(self) -> tuple[str, bool]:
        """Get metadata from the current matmul parent counter."""
        parent_op = "aten.matmul"
        if parent_op in self.node_info.metadata_by_op:
            if parent_op not in self.parent_op_counters:
                self.parent_op_counters[parent_op] = 0
            parent_index = self.parent_op_counters[parent_op]
            parent_metadata_list = self.node_info.metadata_by_op[parent_op]
            if parent_index < len(parent_metadata_list):
                return parent_metadata_list[parent_index], True
        return EmitLoc.make_unknown().to_string(), False

    def _is_view_overflow(self, normalized_op: str, occurrence_index: int) -> bool:
        """Check if a view op would overflow (more dispatched than in FX graph)."""
        if normalized_op != "aten.view":
            return False
        if normalized_op not in self.node_info.metadata_by_op:
            return True  # Not in FX at all
        return occurrence_index >= len(self.node_info.metadata_by_op[normalized_op])

    def _lookup_metadata(
        self, normalized_op: str, occurrence_index: int
    ) -> tuple[str, bool]:
        """
        Look up metadata for an op, with fallback to parent op for decomposed ops.

        Returns:
            tuple of (metadata_string, found_in_fx_graph)
        """
        # If we're in a matmul decomposition, use the cached metadata
        if self._in_matmul_decomp and self._matmul_decomp_metadata is not None:
            return self._matmul_decomp_metadata, True

        # Direct lookup
        if normalized_op in self.node_info.metadata_by_op:
            op_metadata_list = self.node_info.metadata_by_op[normalized_op]
            if occurrence_index < len(op_metadata_list):
                return op_metadata_list[occurrence_index], True

        # Fallback: check if this is a decomposed op that should inherit from parent
        if normalized_op in DECOMPOSITION_PARENT_MAP:
            parent_op = DECOMPOSITION_PARENT_MAP[normalized_op]
            if parent_op in self.node_info.metadata_by_op:
                # Initialize parent counter if needed
                if parent_op not in self.parent_op_counters:
                    self.parent_op_counters[parent_op] = 0

                parent_index = self.parent_op_counters[parent_op]
                parent_metadata_list = self.node_info.metadata_by_op[parent_op]
                if parent_index < len(parent_metadata_list):
                    return parent_metadata_list[parent_index], True

        return EmitLoc.make_unknown().to_string(), False

    def __torch_dispatch__(self, func, types, args=(), kwargs={}):
        res = func(*args, **kwargs)

        actual_target = str(func)
        normalized_op = _normalize_op_name(actual_target)

        # Initialize counter for this op type if needed
        if normalized_op not in self.op_counters:
            self.op_counters[normalized_op] = 0

        occurrence_index = self.op_counters[normalized_op]

        # Detect start of matmul decomposition: view overflow
        # matmul decomposes to: view -> mm -> _unsafe_view
        if (
            self._is_view_overflow(normalized_op, occurrence_index)
            and not self._in_matmul_decomp
        ):
            # This view is overflowing - likely start of matmul decomposition
            # Get matmul metadata and cache it for the sequence
            self._matmul_decomp_metadata, found = self._get_matmul_metadata()
            if found:
                self._in_matmul_decomp = True

        # Look up metadata by op name and occurrence index
        metadata, found = self._lookup_metadata(normalized_op, occurrence_index)

        if not found and self.validate_alignment:
            if normalized_op in self.node_info.metadata_by_op:
                fx_count = len(self.node_info.metadata_by_op[normalized_op])
                self.ops_without_metadata.append(
                    f"{normalized_op}[{occurrence_index}] (overflow: only {fx_count} in FX)"
                )
            else:
                self.ops_without_metadata.append(
                    f"{normalized_op}[{occurrence_index}] (not in FX graph)"
                )

        self._set_metadata(res, metadata)

        # Increment counters
        self.op_counters[normalized_op] += 1
        self.total_dispatched += 1

        # End of matmul decomposition: _unsafe_view is the last op
        if self._in_matmul_decomp and normalized_op == MATMUL_DECOMP_END_OP:
            # Increment the matmul parent counter
            parent_op = "aten.matmul"
            if parent_op not in self.parent_op_counters:
                self.parent_op_counters[parent_op] = 0
            self.parent_op_counters[parent_op] += 1
            # Reset decomposition state
            self._in_matmul_decomp = False
            self._matmul_decomp_metadata = None
        elif normalized_op in DECOMPOSITION_PARENT_MAP:
            # For other decomposed ops (mm, bmm) that aren't in a view->mm->view sequence
            parent_op = DECOMPOSITION_PARENT_MAP[normalized_op]
            if normalized_op == MATMUL_DECOMP_END_OP and parent_op == "aten.matmul":
                if parent_op not in self.parent_op_counters:
                    self.parent_op_counters[parent_op] = 0
                self.parent_op_counters[parent_op] += 1

        return res

    def get_alignment_report(self) -> str:
        """Generate a summary report of alignment between FX graph and runtime."""
        lines = [
            f"Alignment Report (Op-Based Matching):",
            f"  Total FX nodes: {self.node_info.total_fx_nodes}",
            f"  Total dispatched ops: {self.total_dispatched}",
            f"  Unique op types in FX: {len(self.node_info.metadata_by_op)}",
            f"  Unique op types dispatched: {len(self.op_counters)}",
        ]

        if self.ops_without_metadata:
            lines.append(
                f"\nOps without metadata ({len(self.ops_without_metadata)} total):"
            )
            for op in self.ops_without_metadata[:20]:
                lines.append(f"  {op}")
            if len(self.ops_without_metadata) > 20:
                lines.append(f"  ... and {len(self.ops_without_metadata) - 20} more")

        # Show op coverage stats
        lines.append("\nPer-op coverage:")
        for op_name, count in sorted(self.op_counters.items()):
            fx_count = len(self.node_info.metadata_by_op.get(op_name, []))
            status = "✓" if count <= fx_count else f"overflow by {count - fx_count}"
            lines.append(
                f"  {op_name}: dispatched={count}, fx_nodes={fx_count} {status}"
            )

        return "\n".join(lines)

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
                # if tensor.shape == torch.Size([1280]):
                #     print(f"  tensor.shape: {tensor.shape}")
                #     print(f"  module_hierarchy: {module_hierarchy}")
                if tensor.shape == torch.Size([1, 20, 16, 64]):
                    print(f"  tensor.shape: {tensor.shape}")
                    print(f"  module_hierarchy: {module_hierarchy}")
                return True
        except Exception:
            print(f"Error setting metadata - ({module_hierarchy})", flush=True)

        return False
