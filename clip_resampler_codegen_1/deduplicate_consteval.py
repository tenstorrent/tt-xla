#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Script to deduplicate main_const_eval_* functions in main.py.

Uses AST parsing to identify structurally identical functions that differ only
in variable names, then refactors to keep only canonical versions while updating
references (but NOT cache keys, which must remain unique).
"""

import ast
import copy
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class VariableNormalizer(ast.NodeTransformer):
    """
    AST transformer that normalizes variable names to canonical forms.

    Converts names like:
    - utils_DeviceGetter_get_device_42 -> utils_DeviceGetter_get_device_0
    - ttnn_to_device_123 -> ttnn_to_device_0
    - util_create_list_456 -> util_create_list_0

    Does NOT normalize:
    - Module names (ttnn, utils)
    - Function/method names
    - Built-in names
    """

    # Pattern to match variable names with numeric suffixes
    # Must end with _<digits> where digits is at least 1 character
    SUFFIX_PATTERN = re.compile(r"^(.+)_(\d+)$")

    # Names that should never be normalized (modules, builtins, etc.)
    PRESERVE_NAMES = frozenset(
        {
            "ttnn",
            "utils",
            "input",
            "None",
            "True",
            "False",
            # Common attribute chains
            "DeviceGetter",
            "get_device",
            "to_device",
            "to_layout",
            "reshape",
            "repeat",
            "concat",
            "full",
            "Shape",
            "MemoryConfig",
            "TensorMemoryLayout",
            "INTERLEAVED",
            "BufferType",
            "DRAM",
            "Layout",
            "TILE",
            "DataType",
            "BFLOAT16",
        }
    )

    def __init__(self):
        # Track variable definitions in the current function
        self.local_vars: Set[str] = set()
        self.name_mapping: Dict[str, str] = {}
        self.counter: Dict[str, int] = defaultdict(int)

    def reset(self):
        """Reset state for a new function."""
        self.local_vars.clear()
        self.name_mapping.clear()
        self.counter.clear()

    def is_local_variable(self, name: str) -> bool:
        """Check if name looks like a local variable with numeric suffix."""
        if name in self.PRESERVE_NAMES:
            return False
        match = self.SUFFIX_PATTERN.match(name)
        return match is not None

    def normalize_name(self, name: str) -> str:
        """Normalize a variable name by replacing numeric suffix with sequential number."""
        if not self.is_local_variable(name):
            return name

        if name in self.name_mapping:
            return self.name_mapping[name]

        match = self.SUFFIX_PATTERN.match(name)
        if match:
            base = match.group(1)
            # Create canonical name: base_N where N is an incrementing counter
            canonical = f"{base}_{self.counter[base]}"
            self.counter[base] += 1
            self.name_mapping[name] = canonical
            return canonical

        return name

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Normalize Name nodes that are local variables."""
        node.id = self.normalize_name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Keep argument names as-is (they're 'input' typically)."""
        # Don't normalize the 'input' parameter
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Normalize function definition - set name to canonical form."""
        # Normalize the function name to a standard name for comparison
        node.name = "canonical_func"
        # Visit children
        self.generic_visit(node)
        return node


def get_function_signature(func_def: ast.FunctionDef) -> str:
    """
    Generate a canonical signature for a function by normalizing variable names
    and converting back to source code representation.
    """
    # Deep copy the function to avoid modifying the original
    func_copy = copy.deepcopy(func_def)

    # Normalize variable names
    normalizer = VariableNormalizer()
    normalizer.reset()
    normalized = normalizer.visit(func_copy)
    ast.fix_missing_locations(normalized)

    # Convert to a canonical string representation
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def parse_consteval_functions(source: str) -> Dict[str, ast.FunctionDef]:
    """Parse source code and extract main_const_eval_* functions."""
    tree = ast.parse(source)
    functions = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("main_const_eval_"):
                functions[node.name] = node

    return functions


def group_duplicate_functions(
    functions: Dict[str, ast.FunctionDef],
) -> Dict[str, List[str]]:
    """
    Group functions by their canonical signature.

    Returns:
        Dict mapping canonical signature to list of function names with that signature.
    """
    signature_to_funcs: Dict[str, List[str]] = defaultdict(list)

    for func_name, func_def in functions.items():
        sig = get_function_signature(func_def)
        signature_to_funcs[sig].append(func_name)

    return dict(signature_to_funcs)


def extract_func_number(func_name: str) -> int:
    """Extract the numeric ID from a function name like main_const_eval_42."""
    match = re.search(r"_(\d+)$", func_name)
    return int(match.group(1)) if match else 0


def create_refactoring_plan(
    groups: Dict[str, List[str]],
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Create a refactoring plan.

    Returns:
        - mapping: Dict mapping old function names to their canonical replacement
        - canonical_set: Set of function names that are the canonical versions to keep
    """
    mapping = {}
    canonical_set = set()

    for signature, func_names in groups.items():
        # Sort by numeric ID to pick the lowest as canonical
        sorted_names = sorted(func_names, key=extract_func_number)
        canonical = sorted_names[0]
        canonical_set.add(canonical)

        for name in func_names:
            mapping[name] = canonical

    return mapping, canonical_set


def refactor_main_py(
    source: str,
    functions: Dict[str, ast.FunctionDef],
    mapping: Dict[str, str],
    canonical_set: Set[str],
) -> str:
    """
    Refactor main.py:
    1. Remove duplicate function definitions (keep only canonical ones)
    2. Update function REFERENCES in _main() to point to canonical functions
    3. Keep string cache keys unchanged (they must remain unique)
    """
    lines = source.split("\n")

    # Track which lines to remove (duplicate function definitions)
    lines_to_remove = set()

    for func_name, func_def in functions.items():
        if func_name not in canonical_set:
            # Mark all lines of this function for removal
            start_line = func_def.lineno - 1  # 0-indexed
            end_line = func_def.end_lineno  # exclusive
            for i in range(start_line, end_line):
                lines_to_remove.add(i)

    # Build output, removing duplicate functions
    result_lines = []
    i = 0
    while i < len(lines):
        if i in lines_to_remove:
            # Skip this line and subsequent function lines
            while i < len(lines) and i in lines_to_remove:
                i += 1
            continue
        result_lines.append(lines[i])
        i += 1

    # Clean up multiple consecutive blank lines
    cleaned_lines = []
    prev_blank = False
    for line in result_lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank

    # Now ensure two blank lines between function definitions (PEP8)
    final_lines = []
    for i, line in enumerate(cleaned_lines):
        final_lines.append(line)
        # If this line starts a function def and there's only 1 blank before, add another
        if line.startswith("def ") and i > 0:
            # Check if previous line is blank
            if final_lines[-2].strip() == "":
                # Only one blank, need to insert another
                final_lines.insert(-1, "")

    result = "\n".join(final_lines)

    # Now update function REFERENCES (not string cache keys)
    # Pattern: const_X = main_const_eval_Y (where Y is not in quotes)
    for old_name, new_name in mapping.items():
        if old_name != new_name:
            # Replace identifier references only (pattern: = main_const_eval_X at end of assignment)
            # This matches "const_N = main_const_eval_X" but NOT "const_N = "main_const_eval_X""
            result = re.sub(
                rf"(const_\d+\s*=\s*){re.escape(old_name)}(\s*$)",
                rf"\g<1>{new_name}\2",
                result,
                flags=re.MULTILINE,
            )

    return result


def analyze_and_report(source_path: str) -> None:
    """Analyze main.py and report findings."""
    with open(source_path, "r") as f:
        source = f.read()

    print(f"Analyzing {source_path}...")

    functions = parse_consteval_functions(source)
    print(f"Found {len(functions)} main_const_eval_* functions")

    groups = group_duplicate_functions(functions)
    print(f"Found {len(groups)} unique function signatures")

    # Report duplicates
    duplicates_found = 0
    for sig, func_names in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(func_names) > 1:
            duplicates_found += 1
            sorted_names = sorted(func_names, key=extract_func_number)
            canonical = sorted_names[0]
            print(f"\nGroup (canonical: {canonical}):")
            print(
                f"  Duplicates ({len(func_names)} functions): {', '.join(sorted_names)}"
            )

    if duplicates_found == 0:
        print("\nNo duplicates found!")
    else:
        print(f"\n{duplicates_found} groups with duplicates found")
        print(f"Can reduce from {len(functions)} to {len(groups)} functions")

    return functions, groups


def refactor_files(main_path: str, output_suffix: str = "_refactored") -> None:
    """
    Refactor main.py to deduplicate functions.

    Args:
        main_path: Path to main.py
        output_suffix: Suffix for output files (empty string to overwrite originals)
    """
    # Read source
    with open(main_path, "r") as f:
        main_source = f.read()

    # Analyze
    functions = parse_consteval_functions(main_source)
    groups = group_duplicate_functions(functions)
    mapping, canonical_set = create_refactoring_plan(groups)

    print(f"Refactoring: {len(functions)} -> {len(canonical_set)} functions")

    # Generate refactored code
    new_main = refactor_main_py(main_source, functions, mapping, canonical_set)

    # Write output
    main_out = (
        main_path.replace(".py", f"{output_suffix}.py") if output_suffix else main_path
    )

    with open(main_out, "w") as f:
        f.write(new_main)
    print(f"Written: {main_out}")

    # Print mapping for reference
    print("\nFunction mapping:")
    changes = [(old, new) for old, new in mapping.items() if old != new]
    for old, new in sorted(changes, key=lambda x: extract_func_number(x[0])):
        print(f"  {old} -> {new}")


if __name__ == "__main__":
    import os
    import sys

    # Default paths (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(script_dir, "main.py")

    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        # Just analyze and report
        analyze_and_report(main_path)
    elif len(sys.argv) > 1 and sys.argv[1] == "--refactor":
        # Refactor in place
        refactor_files(main_path, output_suffix="")
    elif len(sys.argv) > 1 and sys.argv[1] == "--refactor-copy":
        # Refactor to new files
        refactor_files(main_path, output_suffix="_refactored")
    else:
        print("Usage:")
        print(
            "  python deduplicate_consteval.py --analyze        # Analyze and report duplicates"
        )
        print(
            "  python deduplicate_consteval.py --refactor       # Refactor files in place"
        )
        print(
            "  python deduplicate_consteval.py --refactor-copy  # Create refactored copies"
        )
