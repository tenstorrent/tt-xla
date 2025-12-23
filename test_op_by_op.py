import argparse
from pathlib import Path
import os
import glob

from op_by_op_infra.pydantic_models import OpTest, model_to_dict
from op_by_op_infra.workflow import run_op_by_op_workflow, extract_ops_from_module, execute_extracted_ops

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
                module_ops = extract_ops_from_module(module, origin_model="xd")
                ops.extend(module_ops)
                
            #for op_result in results:
            #    print(f"{op_result.op_name} : {op_result.success}")
    for idx, op_wrapper in enumerate(ops):
        print(f"\n--- Operation {idx + 1}/{len(ops)} ---")
        print(f"op_name: {op_wrapper.op_name}")
        print(f"origin_model: {op_wrapper.origin_model}")
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
    
    return 0


if __name__ == "__main__":
    exit(main())