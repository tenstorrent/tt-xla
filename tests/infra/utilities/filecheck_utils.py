# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
FileCheck utilities for verifying IR patterns in tests.
"""

import subprocess
from pathlib import Path

FILECHECK_PATH = Path("tests/filecheck")


def run_filecheck(
    test_node_id: str, irs_filepath: str, pattern_files: list[str]
) -> dict:
    """
    Run FileCheck on IR files against specified pattern files.

    Looks for IR files in irs_filepath and runs FileCheck against specified pattern files
    from FILECHECK_PATH (tests/filecheck/).

    Args:
        test_node_id: Pytest node ID (e.g., "tests/torch/test_model.py::test_matmul")
        irs_filepath: Directory containing IR files (ttnn.mlir, ttir.mlir, etc.)
        pattern_files: List of pattern filenames to check (e.g., ["concatenate_heads.ttnn.mlir"]).

    Returns:
        dict: Results for each pattern checked:
            {
                "concatenate_heads.ttnn": {"checked": True, "passed": True, "error": None},
                "split_heads.ttir": {"checked": False, "passed": None, "error": "IR file not found"}
            }
    """
    results = {}
    irs_dir = Path(irs_filepath)

    if not irs_dir.exists():
        raise FileNotFoundError(f"IR directory does not exist: {irs_filepath}")

    # Build a mapping of IR types to IR files for the current test
    # We need to match files that correspond to this specific test_node_id
    from tests.infra.utilities.utils import sanitize_test_name

    sanitized_test_id = sanitize_test_name(test_node_id)

    ir_files_map = {}
    for filepath in irs_dir.glob("*.mlir"):
        stem = filepath.stem
        # Check if this IR file corresponds to the current test
        if stem.startswith(sanitized_test_id):
            ir_type = stem.split("_")[-1]
            ir_files_map[ir_type] = filepath

    if not ir_files_map:
        print(f"No IR files found for test '{sanitized_test_id}' in {irs_filepath}")
        # Return results with error for all patterns
        for pattern_filename in pattern_files:
            pattern_stem = Path(pattern_filename).stem
            results[pattern_stem] = {
                "checked": False,
                "passed": None,
                "error": f"No IR files found for test '{sanitized_test_id}' in {irs_filepath}",
            }
        return results

    # Check each specified pattern file
    for pattern_filename in pattern_files:
        pattern_filepath = FILECHECK_PATH / pattern_filename

        # Extract IR type from pattern filename (e.g., "concatenate_heads.ttnn.mlir" -> "ttnn")
        pattern_stem = Path(pattern_filename).stem
        if "." in pattern_stem:
            # Handle "name.ir_type" format
            ir_type = pattern_stem.split(".")[-1]
        else:
            # Handle "name_ir_type" format
            ir_type = pattern_stem.split("_")[-1]

        # Use pattern filename (without .mlir) as result key
        result_key = pattern_stem

        if not pattern_filepath.exists():
            results[result_key] = {
                "checked": False,
                "passed": None,
                "error": f"Pattern file not found: {pattern_filepath}",
            }
            continue

        # Find corresponding IR file
        ir_filepath = ir_files_map.get(ir_type)
        if not ir_filepath:
            results[result_key] = {
                "checked": False,
                "passed": None,
                "error": f"No IR file found for type '{ir_type}' in {irs_filepath}",
            }
            continue

        # Both files exist - run FileCheck
        try:
            _run_filecheck(pattern_filepath, ir_filepath, result_key)
            results[result_key] = {
                "checked": True,
                "passed": True,
                "error": None,
            }
        except subprocess.CalledProcessError as e:
            results[result_key] = {
                "checked": True,
                "passed": False,
                "error": str(e),
            }

    return results


def _run_filecheck(pattern_filepath: Path, ir_filepath: Path, ir_type: str):
    """
    Execute FileCheck command on IR file with pattern file.

    Args:
        pattern_filepath: Path to FileCheck pattern file
        ir_filepath: Path to IR file to check
        ir_type: Type of IR (e.g., "ttnn", "ttir") for logging

    Raises:
        subprocess.CalledProcessError: If FileCheck fails
    """
    print(f"Running FileCheck for {ir_type} IR:")
    print(f"  Pattern: {pattern_filepath}")
    print(f"  IR file: {ir_filepath}")

    result = subprocess.run(
        ["FileCheck", str(pattern_filepath), "--input-file", str(ir_filepath)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(f"\n✗ FileCheck FAILED for {ir_type}:")
        print(f"\nStderr:\n{result.stderr}")
        if result.stdout:
            print(f"\nStdout:\n{result.stdout}")
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )

    print(f"✓ FileCheck PASSED for {ir_type}")
