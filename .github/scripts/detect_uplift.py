# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Uplift detection main script.

Reads the given uplift detection config file and runs each detection script
to determine which uplifts, if any are present in a PR. Outputs a JSON object
describing detected uplifts and the combined test matrix to stdout.

Usage:
    python <this_file> <config_file> <changed_files_file> <diff_file>

Output example:
    {
        "uplifts": ["mlir"],
        "test_matrix": "model-test-extended.json"
    }
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python <this_script> <config_file> <changechanged_files_filed_files> <diff_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])
    changed_files_path = sys.argv[2]
    diff_file_path = sys.argv[3]

    with open(config_path, "r") as f:
        config = json.load(f)

    scripts_dir = config_path.parent / "scripts"

    detected_uplifts = []
    detected_matrices = []

    for entry in config:
        name = entry["name"]
        script = scripts_dir / entry["detector"]
        test_matrix = entry["test_matrix"]

        if not script.exists():
            print(f"Warning: detection script not found: {script}", file=sys.stderr)
            continue

        try:
            result = subprocess.run(
                [str(script), changed_files_path, diff_file_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout.strip():
                print(f"  [{name}] {result.stdout.strip()}", file=sys.stderr)
            if result.returncode == 0:
                print(f"Uplift detected: {name}", file=sys.stderr)
                detected_uplifts.append(name)
                detected_matrices.append(test_matrix)
            else:
                print(f"No uplift: {name}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"Warning: detection script timed out: {script}", file=sys.stderr)
        except Exception as e:
            print(
                f"Warning: error running detection script {script}: {e}",
                file=sys.stderr,
            )

    output = {
        "uplifts": detected_uplifts,
        "test_matrix": ":".join(detected_matrices),
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
