# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
FileCheck utilities for verifying IR patterns in tests.
"""

import subprocess
from pathlib import Path
from tests.infra.utilities.utils import sanitize_test_name

FILECHECK_PATH = Path("tests/filecheck")


def run_filecheck(test_node_id: str, irs_filepath: str) -> dict:
    """
    Run FileCheck on IR files if corresponding pattern files exist.

    Looks for IR files in irs_filepath (e.g., ttnn.mlir, ttir.mlir) and corresponding
    FileCheck pattern files in FILECHECK_PATH named {test_node_id}_{ir_type}.mlir.

    Args:
        test_node_id: Pytest node ID (e.g., "tests/torch/test_model.py::test_matmul")
        irs_filepath: Directory containing IR files (ttnn.mlir, ttir.mlir, etc.)

    Returns:
        dict: Results for each IR type checked:
            {
                "ttnn": {"checked": True, "passed": True, "error": None},
                "ttir": {"checked": False, "passed": None, "error": "Pattern not found"}
            }
    """
    results = {}
    irs_dir = Path(irs_filepath)

    if not irs_dir.exists():
        raise FileNotFoundError(f"IR directory does not exist: {irs_filepath}")

    # Find all .mlir IR files
    ir_files = {}
    for filepath in irs_dir.glob("*.mlir"):
        # Extract IR type from filename
        # Handles both "ttnn.mlir" -> "ttnn" and "test_..._ttir.mlir" -> "ttir"
        stem = filepath.stem
        # Assume IR type is the last underscore-separated component
        ir_type = stem.split('_')[-1]
        ir_files[ir_type] = filepath

    if not ir_files:
        print(f"No .mlir files found in {irs_filepath}")
        return results

    sanitized_test_id = sanitize_test_name(test_node_id)

    # Check each IR type
    for ir_type, ir_filepath in ir_files.items():
        # Look for pattern file: tests/filecheck/{test_node_id}_{ir_type}.mlir
        pattern_filepath = FILECHECK_PATH / f"{sanitized_test_id}_{ir_type}.mlir"

        if not pattern_filepath.exists():
            results[ir_type] = {
                "checked": False,
                "passed": None,
                "error": f"Pattern file not found: {pattern_filepath}",
            }
            continue

        # Both files exist - run FileCheck
        try:
            _run_filecheck(pattern_filepath, ir_filepath, ir_type)
            results[ir_type] = {
                "checked": True,
                "passed": True,
                "error": None,
            }
        except subprocess.CalledProcessError as e:
            results[ir_type] = {
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
