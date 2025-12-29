import pytest
from pathlib import Path
import os
import glob
from typing import List

from op_by_op_infra.pydantic_models import OpTest, model_to_dict
from op_by_op_infra.workflow import run_op_by_op_workflow, extract_ops_from_module, execute_extracted_ops


def pytest_addoption(parser):
    """Add custom command line option for pytest."""
    parser.addoption(
        "--folder",
        action="store",
        default="./collected_irs",
        help="Folder path containing IR files to process (default: ./collected_irs)"
    )
    parser.addoption(
        "--compile-only",
        action="store_true",
        default=False,
        help="Only compile ops without execution (default: False)"
    )


@pytest.fixture
def ir_folder(request):
    """Fixture to get the folder path from command line."""
    return request.config.getoption("--folder")


@pytest.fixture
def compile_only(request):
    """Fixture to get the compile_only flag from command line."""
    return request.config.getoption("--compile_only")


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


def test_op_by_op_compilation(ir_folder, compile_only, record_property):
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

    # Iterate through all files recursively
    print(f"Processing files in: {folder_path}")
    print("-" * 80)
    ops = []

    # Iterate through direct subdirectories
    for subdir in folder_path.iterdir():
        if subdir.is_dir():
            pattern = os.path.join(subdir, "irs", "shlo_compiler*.mlir")
            matches = glob.glob(pattern)
            if not matches:
                pytest.skip(f"No file matching {pattern} with dumped IR found")
            for ir_file_path in matches:
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

                module_ops = extract_ops_from_module(module, origin_model=subdir.name)
                ops.extend(module_ops)

    # Process ops (filtering, etc.)
    ops = filter_duplicate_ops(ops)

    # Record test properties
    record_property("total_ops_before_filtering", len(ops) if not ops else sum(1 for _ in folder_path.iterdir() if _.is_dir()))
    record_property("total_ops", len(ops))
    record_property("ir_folder", str(folder_path))
    record_property("compile_only", compile_only)

    # Execute ops and collect results
    results = execute_extracted_ops(ops, compile_only=compile_only)

    # Track statistics
    successful_ops = []
    failed_ops = []

    for op_result in results:
        print(f"{op_result.op_name} : {op_result.success}")

        if op_result.success:
            successful_ops.append(op_result.op_name)
        else:
            failed_ops.append(op_result.op_name)

        # Record individual op result
        record_property(f"op_{op_result.op_name}_success", op_result.success)
        if hasattr(op_result, 'error_message') and op_result.error_message:
            record_property(f"op_{op_result.op_name}_error", op_result.error_message)

    # Record summary statistics
    record_property("successful_ops_count", len(successful_ops))
    record_property("failed_ops_count", len(failed_ops))
    record_property("success_rate", len(successful_ops) / len(results) if results else 0)

    if successful_ops:
        record_property("successful_ops", ", ".join(successful_ops))
    if failed_ops:
        record_property("failed_ops", ", ".join(failed_ops))

    # Assert that all ops succeeded
    assert len(failed_ops) == 0, f"Failed ops: {', '.join(failed_ops)}"