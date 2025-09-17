# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import sys
from pathlib import Path


def map_shared_runner(entry):
    """
    Map runs-on value to shared runner equivalent if shared-runners is enabled.

    Args:
        entry: Dictionary representing a test matrix entry

    Returns:
        Updated entry (copy if modified, original if not)
    """

    shared_runner_mapping = {
        "n150": "tt-ubuntu-2204-n150-stable",
        "n300": "tt-ubuntu-2204-n300-stable",
        "wormhole_b0": "tt-ubuntu-2204-n300-stable",
        "p150": "tt-ubuntu-2204-p150b-stable",
        "n300-llmbox": "tt-ubuntu-2204-n300-llmbox-stable",
    }

    if entry.get("shared-runners") == "true" or entry.get("shared-runners") is True:
        runs_on = entry.get("runs-on")
        if runs_on in shared_runner_mapping:
            entry["runs-on"] = shared_runner_mapping[runs_on]
    return entry


def expand_parallel_entry(entry, expanded_matrix):
    """
    Expand entry with parallel_groups into multiple entries with group_id.

    Args:
        entry: Dictionary representing a test matrix entry
        expanded_matrix: List to append the expanded entries to
    """
    if "parallel_groups" in entry:
        parallel_groups = entry["parallel_groups"]

        for group_id in range(1, parallel_groups + 1):
            new_entry = entry.copy()
            new_entry["group_id"] = group_id
            expanded_matrix.append(new_entry)
    else:
        expanded_matrix.append(entry)


def process_test_matrix(matrix_file_path):
    """
    Process test matrix by expanding parallel groups and mapping shared runners.

    For each entry with 'parallel_groups': n, creates n total entries
    (including the original) where each has a 'group_id' starting from 1.

    Also handles 'shared-runners': true by mapping runs-on values to their
    corresponding shared runner equivalents.
    """

    # Read the input JSON file
    with open(matrix_file_path, "r") as f:
        matrix = json.load(f)

    if not isinstance(matrix, list):
        raise ValueError("Expected JSON file to contain an array at the root level")

    expanded_matrix = []

    for entry in matrix:
        map_shared_runner(entry)

        expand_parallel_entry(entry, expanded_matrix)

    return expanded_matrix


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <test-matrix-json-file>", file=sys.stderr)
        sys.exit(1)

    matrix_file_path = sys.argv[1]

    if not Path(matrix_file_path).exists():
        print(f"Error: File '{matrix_file_path}' not found", file=sys.stderr)
        sys.exit(1)

    try:
        expanded_matrix = process_test_matrix(matrix_file_path)

        print(json.dumps(expanded_matrix, indent=2))

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{matrix_file_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
