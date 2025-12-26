# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example script that runs the op-by-op workflow.

This example allows the user to:
1. Read input MLIR module from a file
2. Choose between different op-by-op workflow types

Requires:
    # Building with TTMLIR_ENABLE_RUNTIME=ON and TTMLIR_ENABLE_STABLEHLO=ON
    cmake -G Ninja -B build -DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_STABLEHLO=ON
    cmake --build build

    # Having system descriptor saved
    ttrt query --save-artifacts

Usage Examples:
    # Generate JSON report with pytest
    pytest -svv test/python/op_by_op/op_by_op_read_ir_from_file.py::test_op_by_op_inference_from_file --json-report --json-report-file=report.json

    # Use a custom MLIR file path
    export OP_BY_OP_MLIR_FILE_PATH=/path/to/your/model.mlir
    pytest -svv test/python/op_by_op/op_by_op_read_ir_from_file.py::test_op_by_op_inference_from_file --json-report --json-report-file=report.json

    # Try different workflow types (currently unsupported)
    export OP_BY_OP_WORKFLOW_TYPE=compile_split_and_execute
    pytest -svv test/python/op_by_op/op_by_op_read_ir_from_file.py::test_op_by_op_inference_from_file --json-report --json-report-file=report.json

    # Enable compile-only mode
    export OP_BY_OP_COMPILE_ONLY=true
    pytest -svv test/python/op_by_op/op_by_op_read_ir_from_file.py::test_op_by_op_inference_from_file --json-report --json-report-file=report.json

Workflow Types:
    - split_and_execute: Split module into individual ops, then execute (default)
    - compile_split_and_execute: Compile the full module first, then split and execute
    - split_compile_split_and_execute: Split first, compile each op individually, split then execute

Environment Variables:
    - OP_BY_OP_MLIR_FILE_PATH: Path to IR file (default: test/python/op_by_op/example_shlo_ir.mlir)
    - OP_BY_OP_WORKFLOW_TYPE: Workflow execution strategy (default: split_and_execute)
    - OP_BY_OP_COMPILE_ONLY: Only compile operations without executing them (default: false)

Note:
    - File should contain one MLIR module in StableHLO dialect
"""

import os
from pathlib import Path

from op_by_op_infra import workflow_internal, workflow
from op_by_op_infra.workflow import extract_ops_from_module, execute_extracted_ops
from op_by_op_infra.pydantic_models import OpTest, model_to_dict


def run_op_by_op_workflow():
    file_path = _get_mlir_file_path()

    module = _read_mlir_file(file_path)
    if module is None:
        raise AssertionError("Failed to read MLIR module")

    return extract_ops_from_module(
        module,
        origin_model="xd",
    )


def _get_mlir_file_path() -> Path:
    env_path = os.getenv("OP_BY_OP_MLIR_FILE_PATH")

    if env_path:
        user_path = Path(env_path)
        print(f"INFO: Using user-specified MLIR file: {user_path}")
        return user_path

    default_path = Path(__file__).parent / "example_shlo_ir.mlir"
    print(f"INFO: Using default MLIR file: {default_path}")
    print(
        "INFO: To use a different file, set: export OP_BY_OP_MLIR_FILE_PATH=/path/to/your/file.mlir"
    )

    return default_path


def _read_mlir_file(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content == "":
            raise ValueError(f"IR file is empty: {file_path}")

        print(f"INFO: Successfully read IR file ({len(content)} characters)")
        return content

    except Exception as e:
        raise RuntimeError(f"Error reading IR file {file_path}: {e}")


def test_op_by_op_inference_from_file(record_property):
    """
    Test function that generates JSON report with operation information.

    This test function runs the op-by-op workflow and records properties
    for each operation result, similar to the frontend workflow approach.
    When run with pytest --json-report --json-report-file=report.json,
    it will generate a JSON file with detailed operation information.
    """

    # Execute the workflow
    ops = run_op_by_op_workflow()

    # Debug: Print information about each OpWrapper
    print(f"\n{'='*80}")
    print(f"DEBUG: Found {len(ops)} operations")
    print(f"{'='*80}\n")
    
    for idx, op_wrapper in enumerate(ops):
        print(f"\n--- Operation {idx + 1}/{len(ops)} ---")
        print(f"op_name: {op_wrapper.op_name}")
        print(f"origin_model: {', '.join(op_wrapper.origin_model) if op_wrapper.origin_model else 'None'}")
        if len(op_wrapper.origin_model) > 1:
            print(f"*** SHARED OPERATION: Found in {len(op_wrapper.origin_model)} models ***")
        print(f"op_string: {op_wrapper.op_string if op_wrapper.op_string else 'None'}...")
        print(f"func_op_string: {op_wrapper.func_op_string if op_wrapper.func_op_string else 'None'}...")
        print(f"Number of operands: {len(op_wrapper.operands)}")
        for i, operand in enumerate(op_wrapper.operands):
            print(f"  Operand {i}: name={operand.name}, type={operand.type}")
        print(f"Number of results: {len(op_wrapper.results)}")
        for i, result in enumerate(op_wrapper.results):
            print(f"  Result {i}: name={result.name}, type={result.type}")
        if op_wrapper.attributes is not None:
          print("Attributes:")
          for attr in op_wrapper.attributes:
            print(f"  {attr.name} = {attr.attr}")
        else:
            print("Attributes: None")
    
    print(f"\n{'='*80}")
    print(f"DEBUG: Completed processing all operations")
    print(f"{'='*80}\n")

    results = execute_extracted_ops(ops, True)

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