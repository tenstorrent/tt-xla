# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Op-by-op test that processes StableHLO IR files from a folder.

This test allows the user to:
1. Process all .mlir files from a specified folder
2. Extract operations from each IR file
3. Filter operations by whitelist/blacklist and remove duplicates
4. Execute operations individually (with optional compile-only and debug-print modes)

Requires:
    # Building with -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON
    cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON
    cmake --build build

    # Having system descriptor saved
    ttrt query --save-artifacts

Usage Examples:
    # Run with a folder containing .mlir files, generate JSON report
    pytest -svv tests/op_by_op/op_by_op_test.py::test_op_by_op --folder=ir_files --json-report --json-report-file=report.json

    # Filter ops using whitelist
    pytest -svv tests/op_by_op/op_by_op_test.py::test_op_by_op --folder=ir_files --whitelist="stablehlo.add,stablehlo.multiply" --json-report --json-report-file=report.json

    # Match files based on prefix within model directories
    # Example: IRs were gathered using --dump-irs option in tests/runner/test_models.py (filepath: collected_irs/model_name/irs/shlo_compiler_*.mlir)
    pytest -svv tests/op_by_op/op_by_op_test.py::test_op_by_op --folder=collected_irs --ir-file-prefix="irs/shlo_compiler" --json-report --json-report-file=report.json
    
Command Line Arguments:
    --folder: Path to folder containing .mlir files (required)
    --ir-file-prefix: Path pattern within each model directory for filtering files and extracting model names
                      Format: "dir1/dir2/.../filename_prefix" where model name = parent directory before dir1
                      Example: "irs/shlo_compiler" matches folder/.../model_name/irs/shlo_compiler_*.mlir (model_name extracted automatically)
                      Empty (default) processes all .mlir files with immediate parent directory as model name
    --compile-only: Only compile operations without executing them (optional, default: False)
    --debug-print: Enable debug printing during operation execution (optional, default: False)
    --whitelist: Comma-separated list of ops to test (if set, only these ops are tested)
    --blacklist: Comma-separated list of ops to skip (ignored if whitelist is set)

Note:
    - Each file should contain a MLIR module in StableHLO dialect
"""

from pathlib import Path
from typing import List

import pytest
from op_by_op_infra.pydantic_models import OpTest, model_to_dict
from op_by_op_infra.workflow import execute_extracted_ops, extract_ops_from_module


@pytest.fixture
def whitelist(request) -> List[str]:
    cli_whitelist = request.config.getoption("--whitelist", default=None)
    if cli_whitelist:
        return [op.strip() for op in cli_whitelist.split(",") if op.strip()]
    return []


@pytest.fixture
def blacklist(request) -> List[str]:
    cli_blacklist = request.config.getoption("--blacklist", default=None)
    if cli_blacklist:
        return [op.strip() for op in cli_blacklist.split(",") if op.strip()]
    return []


def match_and_extract_model_name(file_path: Path, ir_file_prefix: str) -> str | None:
    """
    Check if file matches the IR file prefix pattern and extract model name.
    Returns model name if file matches, None otherwise.
    """
    if not ir_file_prefix:
        return file_path.parent.name

    parts = ir_file_prefix.split("/")
    file_prefix = parts[-1]
    dir_parts = parts[:-1]

    if not file_path.name.startswith(file_prefix):
        return None

    if not dir_parts:
        return file_path.parent.name

    path_parts = list(file_path.parts)
    for i in range(len(path_parts) - len(dir_parts)):
        if path_parts[i : i + len(dir_parts)] == dir_parts:
            if i > 0:
                return path_parts[i - 1]
            return dir_parts[0]

    return None


def filter_and_deduplicate_ops(
    ops: List, whitelist: List[str] = None, blacklist: List[str] = None
) -> List:
    """Filter operations by whitelist/blacklist and deduplicate by op_string."""
    seen_strings = {}
    unique_ops = []

    for op_wrapper in ops:
        op_string = op_wrapper.op_string
        op_name = op_wrapper.op_name

        if whitelist:
            if op_name not in whitelist:
                continue
        elif blacklist:
            if op_name in blacklist:
                continue

        if op_string and op_string in seen_strings:
            first_occurrence = seen_strings[op_string]
            for model in op_wrapper.origin_model:
                first_occurrence.add_origin_model(model)
            continue

        unique_ops.append(op_wrapper)
        if op_string:
            seen_strings[op_string] = op_wrapper

    return unique_ops


def test_op_by_op(request, whitelist, blacklist, record_property):
    """Test op-by-op compilation from IR files."""
    ir_folder = request.config.getoption("--folder")
    compile_only = request.config.getoption("--compile-only")
    debug_print = request.config.getoption("--debug-print")
    ir_file_prefix = request.config.getoption("--ir-file-prefix")

    folder_path = Path(ir_folder)

    if not folder_path.exists():
        pytest.fail(f"Error: Folder '{folder_path}' does not exist")

    if not folder_path.is_dir():
        pytest.fail(f"Error: '{folder_path}' is not a directory")

    all_mlir_files = list(folder_path.rglob("*.mlir"))
    matched_files = [
        (f, model_name)
        for f in all_mlir_files
        if (model_name := match_and_extract_model_name(f, ir_file_prefix)) is not None
    ]

    if not matched_files:
        if all_mlir_files:
            pytest.skip(
                f"No .mlir files matching IR file prefix '{ir_file_prefix}' found in {folder_path}"
            )
        else:
            pytest.skip(f"No .mlir files found in {folder_path}")

    ops = []
    for ir_file_path, origin_model in matched_files:
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

        module_ops = extract_ops_from_module(module, origin_model=origin_model)
        ops.extend(module_ops)

    filtered_ops = filter_and_deduplicate_ops(
        ops, whitelist=whitelist, blacklist=blacklist
    )

    record_property("total_ops_before_filtering", len(ops))
    record_property("total_ops_after_filtering", len(filtered_ops))
    record_property("ir_folder", str(folder_path))
    record_property("ir_file_prefix", ir_file_prefix if ir_file_prefix else None)
    record_property("compile_only", compile_only)
    record_property("whitelist", whitelist if whitelist else None)
    record_property("blacklist", blacklist if blacklist else None)

    results = execute_extracted_ops(
        filtered_ops, compile_only=compile_only, debug_print=debug_print
    )

    for result in results:
        record_property(f"OpTest model for: {result.op_name}", model_to_dict(result))

    successful_operations = sum(1 for r in results if r.success)
    failed_operations = sum(1 for r in results if not r.success)
    record_property("successful_operations", successful_operations)
    record_property("failed_operations", failed_operations)

    assert (
        failed_operations == 0
    ), f"Test failed: {failed_operations} operation(s) failed out of {len(results)} total operations"
