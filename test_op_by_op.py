import argparse
from pathlib import Path
import os
import glob
import re
from typing import List

from op_by_op_infra.pydantic_models import OpTest, model_to_dict
from op_by_op_infra.workflow import run_op_by_op_workflow, extract_ops_from_module, execute_extracted_ops
from op_by_op_infra.mlir_module_splitter import MLIRModuleSplitter


def normalize_op_string(op_string: str) -> str:
    """
    Normalize an op string by replacing SSA value names (e.g., %arg0, %0, %result)
    with placeholders. This allows comparing operations by structure rather than
    specific variable names.
    """
    if not op_string:
        return ""
    # Replace SSA values like %arg0, %0, %123, %result with placeholder
    normalized = re.sub(r'%[a-zA-Z_][a-zA-Z0-9_]*', '%VAR', op_string)
    normalized = re.sub(r'%\d+', '%VAR', normalized)
    return normalized


def filter_duplicate_ops(ops: List, verbose: bool = True) -> List:
    """
    Filter duplicate operations from the list.

    Two operations are considered duplicates if they have the same normalized
    op_string (i.e., same structure with different variable names).

    When a duplicate is found, if it appears in a different model, that model
    is added to the origin_model list of the first occurrence.

    Returns a list with only unique operations.
    """
    seen_normalized_strings = {}
    unique_ops = []
    verbose = False

    for op_wrapper in ops:
        # Normalize op_string for structure-based comparison
        normalized_string = normalize_op_string(op_wrapper.op_string)

        # Check if we've seen this op before
        if normalized_string and normalized_string in seen_normalized_strings:
            # Get the first occurrence of this op
            first_occurrence = seen_normalized_strings[normalized_string]

            # Add the current op's origin_model(s) to the first occurrence if not already present
            for model in op_wrapper.origin_model:
                first_occurrence.add_origin_model(model)

            if verbose:
                print(f"Filtering duplicate op:")
                print(op_wrapper.op_string)
                print(normalized_string)
            continue

        # This is a unique op, add it
        unique_ops.append(op_wrapper)
        if normalized_string:
            seen_normalized_strings[normalized_string] = op_wrapper

    return unique_ops


def process_ops(ops: List, enable_filtering: bool = True) -> List:
    """
    Process collected ops with various transformations.
    This function serves as an extensible processing pipeline.

    Args:
        ops: List of collected operations
        enable_filtering: Whether to filter duplicate ops

    Returns:
        Processed list of operations
    """
    processed_ops = ops

    print(f"\n{'='*80}")
    print(f"Processing {len(ops)} collected operations...")
    print(f"{'='*80}\n")

    # Filter duplicates
    if enable_filtering:
        print("Filtering duplicate operations...")
        processed_ops = filter_duplicate_ops(processed_ops, verbose=True)
        print(f"Filtered: {len(ops)} -> {len(processed_ops)} operations\n")

    # Future processing options can be added here:
    # - Sorting by complexity
    # - Grouping by op type
    # - Filtering by specific criteria
    # - etc.

    return processed_ops


def main():
    parser = argparse.ArgumentParser(
        description="Process files in a folder (including nested subfolders)"
    )
    parser.add_argument(
        '--folder',
        required=True,
        help='Folder path containing files to process'
    )
    args = parser.parse_args()

    folder_path = Path(args.folder)

    # Check if folder exists
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return 1

    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        return 1

    # Iterate through all files recursively
    print(f"Processing files in: {folder_path}")
    print("-" * 80)
    ops = []
    attributes = None  # Store attributes from the first module

    # Iterate through direct subdirectories
    for subdir in folder_path.iterdir():
        if subdir.is_dir():
            pattern = os.path.join(subdir, "irs", "shlo_compiler*.mlir")
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"No file matching {pattern} with dumped IR found")
            for ir_file_path in matches:
                try:
                    with open(ir_file_path, "r") as f:
                        module = f.read()
                except (FileNotFoundError, IOError, OSError) as e:
                    pytest.fail(
                        f"Op-by-op test failed because IR file couldn't be read.\n"
                        f"Test: {nodeid}\n"
                        f"File: {ir_file_path}"
                    )
                print(f"Processing IR file: {ir_file_path}")

                # Use MLIRModuleSplitter to extract ops and attributes
                splitter = MLIRModuleSplitter()
                module_ops = splitter.split(module, origin_model=subdir.name)
                ops.extend(module_ops)

                # Store attributes from the first module (assuming all modules have same attributes)
                if attributes is None:
                    attributes = splitter.attributes
                    if attributes is not None:
                        print("Extracted module attributes:")
                        for attr in attributes:
                            print(f"  {attr.name} = {attr.attr}")

            #for op_result in results:
            #    print(f"{op_result.op_name} : {op_result.success}")

    # Process ops (filtering, etc.)
    ops = process_ops(ops, enable_filtering=True)

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

    # Print module attributes (same for all ops from the same module)
    if attributes is not None:
        print("\n--- Module Attributes (shared by all ops) ---")
        for attr in attributes:
            print(f"  {attr.name} = {attr.attr}")
    else:
        print("\n--- Module Attributes: None ---")
    
    print(f"\n{'='*80}")
    print(f"DEBUG: Completed processing all operations")
    print(f"{'='*80}\n")

    # Example: If you want to execute the extracted ops, you can do:
    #results = execute_extracted_ops(ops, attributes=attributes, compile_only=False)
    #for op_result in results:
    #    print(f"{op_result.op_name} : {op_result.success}")

    return 0


if __name__ == "__main__":
    exit(main())