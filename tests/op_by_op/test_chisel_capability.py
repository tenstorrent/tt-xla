# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test to verify that the explorer build can run pytest with Chisel enabled and
produces a non-empty Chisel report.

It spawns a pytest subprocess with `--enable-chisel` against a small op test and
checks that the `chisel_context` fixture wrote a non-empty JSON report.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

# Small, fast op test used to exercise the Chisel session.
TEST_FILE = "tests/torch/ops/test_mul.py"
TEST_NAME = "test_mul"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    target = f"{repo_root / TEST_FILE}::{TEST_NAME}"

    with tempfile.TemporaryDirectory() as workdir:
        workdir = Path(workdir)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-svv",
            "--enable-chisel",
            target,
        ]
        print(f"Running: {' '.join(cmd)} (cwd={workdir})")
        result = subprocess.run(cmd, cwd=workdir)
        if result.returncode != 0:
            print(f"pytest subprocess failed with exit code {result.returncode}")
            return result.returncode

        results_dir = workdir / "chisel_results"
        if not results_dir.is_dir():
            print(f"Expected Chisel results directory not found: {results_dir}")
            return 1

        reports = list(results_dir.glob("*.jsonl"))
        if not reports:
            print(f"No Chisel report produced in {results_dir}")
            return 1

        found_numerics = False
        for report in reports:
            print(f"Found Chisel report: {report}")

            text = report.read_text()
            if "numerics" in text:
                found_numerics = True
                print(f"Found 'numerics' records in {report}")

        if not found_numerics:
            print("No 'numerics' records found in any Chisel report")
            return 1

    print("Chisel capability test passed: non-empty report with numerics produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
