# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Op-by-op test that processes MLIR IR files from a folder.

This test allows the user to:
1. Process all .mlir files recursively from a specified folder
2. Extract operations from each IR file
3. Filter duplicate operations
4. Execute operations individually (with optional compile-only mode)

Requires:
    # Building with -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON
    cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON
    cmake --build build

    # Having system descriptor saved
    ttrt query --save-artifacts

Usage Examples:
    # Run with a folder containing .mlir files, generate JSON report
    pytest -svv tests/op_by_op_wip/test_op_by_op.py::test_op_by_op --folder=/path/to/ir/folder --json-report --json-report-file=report.json

    # Enable compile-only mode (skip execution)
    pytest -svv tests/op_by_op_wip/test_op_by_op.py::test_op_by_op --folder=/path/to/ir/folder --compile-only --json-report --json-report-file=report.json

Command Line Arguments:
    --folder: Path to folder containing .mlir files (required)
    --compile-only: Only compile operations without executing them (optional)

Note:
    - All .mlir files in the folder (including subdirectories) will be processed
    - Each file should contain a MLIR module in StableHLO dialect
    - Duplicate operations (same op signature) are filtered and consolidated
    - Operations from different models are tracked via origin_model field
"""

import pytest
from pathlib import Path
import os
import glob
from typing import List

from op_by_op_infra.pydantic_models import OpTest, model_to_dict
from op_by_op_infra.workflow import run_op_by_op_workflow, extract_ops_from_module, execute_extracted_ops


@pytest.fixture
def ir_folder(request):
    """Fixture to get the folder path from command line."""
    return request.config.getoption("--folder")


@pytest.fixture
def compile_only(request):
    """Fixture to get the compile-only flag from command line."""
    return request.config.getoption("--compile-only")


def filter_duplicate_ops(ops: List) -> List:
    """
    Filter duplicate operations from the list.

    Two operations are considered duplicates if they have the same
    op_string (name, attributes and operand/result types/shapes).

    When a duplicate is found, if it appears in a different model, that model
    is added to the origin_model list of the first occurrence.
    """
    seen_normalized_strings = {}
    unique_ops = []

    for op_wrapper in ops:
        normalized_string = op_wrapper.op_string

        # Check if we've seen this op before
        if normalized_string and normalized_string in seen_normalized_strings:
            # Add the current op's origin_model(s) to the first occurrence of this op (if not already present)
            first_occurrence = seen_normalized_strings[normalized_string]
            for model in op_wrapper.origin_model:
                first_occurrence.add_origin_model(model)

            continue

        # This is a unique op, add it
        unique_ops.append(op_wrapper)
        if normalized_string:
            seen_normalized_strings[normalized_string] = op_wrapper

    return unique_ops


def test_op_by_op(ir_folder, compile_only, record_property):
    """
    Test op-by-op compilation from IR files.

    This test processes IR files from the specified folder, extracts operations,
    filters duplicates, and executes them. Results are recorded using record_property
    for JSON reporting.
    """
    folder_path = Path(ir_folder)

    if not folder_path.exists():
        pytest.fail(f"Error: Folder '{folder_path}' does not exist")

    if not folder_path.is_dir():
        pytest.fail(f"Error: '{folder_path}' is not a directory")

    # Iterate through all .mlir files recursively at any depth
    print(f"Processing files in: {folder_path}")
    print("-" * 80)
    ops = []

    # Find all .mlir files recursively
    mlir_files = list(folder_path.rglob("*.mlir"))

    if not mlir_files:
        pytest.skip(f"No .mlir files found in {folder_path}")

    for ir_file_path in mlir_files:
        try:
            with open(ir_file_path, "r") as f:
                module = f.read()
        except (FileNotFoundError, IOError, OSError) as e:
            pytest.fail(
                f"Op-by-op test failed because IR file couldn't be read.\n"
                f"File: {ir_file_path}\n"
                f"Error: {e}"
            )
        print(f"Processing IR file: {ir_file_path}")

        # Use parent directory name as origin_model
        origin_model = ir_file_path.parent.name
        module_ops = extract_ops_from_module(module, origin_model=origin_model)
        ops.extend(module_ops)

    # Process ops (filtering, etc.)
    filtered_ops = filter_duplicate_ops(ops)

    # Record test properties
    record_property("total_ops_before_filtering", len(ops))
    record_property("total_ops", len(filtered_ops))
    record_property("ir_folder", str(folder_path))
    record_property("compile_only", compile_only)

    # Execute ops and collect results
    results = execute_extracted_ops(filtered_ops, compile_only=compile_only)

    for result in results:
        record_property(f"OpTest model for: {result.op_name}", model_to_dict(result))

    # Also record summary information
    record_property("total_operations", len(results))
    successful_operations = sum(1 for r in results if r.success)
    failed_operations = sum(1 for r in results if not r.success)
    record_property("successful_operations", successful_operations)
    record_property("failed_operations", failed_operations)

    # Fail the test if there are any failed operations
    assert (
        failed_operations == 0
    ), f"Test failed: {failed_operations} operation(s) failed out of {len(results)} total operations"