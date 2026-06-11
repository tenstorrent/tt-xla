#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AST-based pytest marker checker - no imports, no hardware modules loaded.
Statically parses test files and filters by marker expression.
"""

import ast
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple


def extract_markers_from_file(file_path: Path) -> List[Tuple[str, Set[str]]]:
    """
    Parse a test file and extract test functions with their markers.

    Returns:
        List of (test_name, {marker_names}) tuples
    """
    with open(file_path, "r") as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
            return []

    results = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (
            node.name.startswith("test_")
            or any(
                isinstance(d, ast.Name) and d.id == "test"
                for d in node.decorator_list
                if isinstance(d, ast.Name)
            )
        ):
            markers = set()

            # Extract markers from decorators
            for decorator in node.decorator_list:
                marker_name = _extract_marker_name(decorator)
                if marker_name:
                    markers.add(marker_name)

            results.append((node.name, markers))

    return results


def _extract_marker_name(decorator: ast.expr) -> Optional[str]:
    """Extract marker name from a pytest.mark decorator."""

    # Handle: @pytest.mark.name
    if isinstance(decorator, ast.Attribute):
        if isinstance(decorator.value, ast.Attribute):
            if (
                isinstance(decorator.value.value, ast.Name)
                and decorator.value.value.id == "pytest"
            ):
                if decorator.value.attr == "mark":
                    return decorator.attr

    # Handle: @mark.name (if mark is imported)
    elif isinstance(decorator, ast.Attribute):
        if isinstance(decorator.value, ast.Name) and decorator.value.id == "mark":
            return decorator.attr

    # Handle: @pytest.mark.name(args)
    if isinstance(decorator, ast.Call):
        if isinstance(decorator.func, ast.Attribute):
            if isinstance(decorator.func.value, ast.Attribute):
                if (
                    isinstance(decorator.func.value.value, ast.Name)
                    and decorator.func.value.value.id == "pytest"
                ):
                    if decorator.func.value.attr == "mark":
                        return decorator.func.attr
            elif (
                isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "mark"
            ):
                return decorator.func.attr

    return None


def evaluate_marker_expression(markers: Set[str], expression: str) -> bool:
    """
    Evaluate a pytest marker expression against a set of markers.

    Supports: 'and', 'or', 'not', parentheses
    Example: "(nightly or push) and not (single_device or dual_chip)"
    """
    import re

    # Build a boolean expression by replacing marker names with presence checks
    eval_expr = expression

    # Extract all potential marker names from the expression
    all_markers = _extract_marker_names_from_expr(expression)

    # Sort by length (descending) to avoid partial replacements
    sorted_markers = sorted(all_markers, key=len, reverse=True)

    # Replace each marker name with True/False using word boundaries
    for marker in sorted_markers:
        # Use word boundaries to avoid replacing parts of other words
        eval_expr = re.sub(
            r"\b" + re.escape(marker) + r"\b", str(marker in markers), eval_expr
        )

    try:
        return eval(eval_expr)
    except Exception as e:
        print(f"Error evaluating expression '{expression}': {e}", file=sys.stderr)
        return False


def _extract_marker_names_from_expr(expression: str) -> Set[str]:
    """Extract marker names from an expression string."""
    import re

    # Replace operators and parentheses with spaces (preserving word boundaries)
    # First, add spaces around operators and parentheses
    expr_with_spaces = re.sub(r"(\(|\))", r" \1 ", expression)
    expr_with_spaces = re.sub(r"\b(and|or|not)\b", r" \1 ", expr_with_spaces)

    # Now extract identifiers
    names = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr_with_spaces))

    # Remove keywords
    for keyword in ("and", "or", "not"):
        names.discard(keyword)

    return names


def main():
    test_dir = Path("tests/torch")

    # Marker expression to filter by
    marker_expression = "(nightly or push) and not (single_device or dual_chip or llmbox or galaxy or bh_galaxy or multi_host_cluster)"

    if len(sys.argv) > 1:
        marker_expression = sys.argv[1]

    print(f"Searching for tests matching: {marker_expression}\n", file=sys.stderr)

    matching_tests = []
    all_tests = []

    # Recursively find all test files
    test_files = sorted(test_dir.rglob("test_*.py"))

    if not test_files:
        print(f"No test files found in {test_dir}", file=sys.stderr)
        return 0

    for test_file in test_files:
        tests = extract_markers_from_file(test_file)

        for test_name, markers in tests:
            all_tests.append((test_file, test_name, markers))

            if evaluate_marker_expression(markers, marker_expression):
                matching_tests.append((test_file, test_name, markers))

    # Print results
    print(
        f"Matches: {len(matching_tests)} / {len(all_tests)} total tests\n",
        file=sys.stderr,
    )

    cwd = Path.cwd()
    if matching_tests:
        current_file = None
        for test_file, test_name, markers in sorted(matching_tests):
            # Make path relative to cwd if possible
            try:
                rel_path = test_file.relative_to(cwd)
            except ValueError:
                rel_path = test_file

            if test_file != current_file:
                print(f"{rel_path}")
                print(f"FILE: {rel_path}", file=sys.stderr)
                current_file = test_file

            marker_str = ", ".join(sorted(markers)) if markers else "(no markers)"
            print(f"   - {test_name} [{marker_str}]", file=sys.stderr)
    else:
        print("SUCCESS: No tests match the expression.", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    main()
    sys.exit(0)
