# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Uplift detection main script.

Reads the given uplift detection config file and checks the PR diff for
changes matching each entry's path and regex pattern. Outputs JSON to stdout:

    {
      "uplifts": ["uplift1", "uplift2"],
      "test_matrices": ["matrix1.json", "matrix2.json"]
    }

Usage:
    python <this_file> <config_file> <changed_files_file> <diff_file>
"""

import json
import re
import sys
from pathlib import Path


def check_uplift(entry, changed_files, diff_content):
    """Check if an uplift is detected based on config entry."""
    name = entry["name"]
    path = entry["path"]
    regex = entry["regex"]

    if path not in changed_files:
        print(f"No uplift: {name}", file=sys.stderr)
        return False

    # Extract the diff section for the specific file
    escaped_path = re.escape(path)
    section_pattern = (
        rf"^diff --git a/{escaped_path} b/{escaped_path}\n(.*?)(?=^diff --git |\Z)"
    )
    section_match = re.search(section_pattern, diff_content, re.MULTILINE | re.DOTALL)

    if not section_match:
        print(f"No uplift: {name}", file=sys.stderr)
        return False

    file_diff = section_match.group(0)
    if re.search(regex, file_diff, re.MULTILINE):
        print(
            f"Detected change matching /{regex}/ in {path}",
            file=sys.stderr,
        )
        print(f"Uplift detected: {name}", file=sys.stderr)
        return True

    print(f"No uplift: {name}", file=sys.stderr)
    return False


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python <this_script> <config_file> <changed_files_file> <diff_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])
    changed_files_path = sys.argv[2]
    diff_file_path = sys.argv[3]

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(changed_files_path, "r") as f:
        changed_files = {line.strip() for line in f if line.strip()}

    with open(diff_file_path, "r") as f:
        diff_content = f.read()

    result = {"uplifts": [], "test_matrices": []}

    for entry in config:
        if check_uplift(entry, changed_files, diff_content):
            result["uplifts"].append(entry["name"])
            result["test_matrices"].append(entry["test_matrix"])

    print(json.dumps(result))


if __name__ == "__main__":
    main()
