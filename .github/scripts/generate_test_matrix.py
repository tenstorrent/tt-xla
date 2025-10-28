# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import sys
from pathlib import Path


def map_runner_name(entry):
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
            # Preserve the original runs-on so downstream consumers
            # (e.g. --arch for test_models.py) can access non mapped arch.
            entry["runs-on-original"] = runs_on
            entry["runs-on"] = shared_runner_mapping[runs_on]
        else:
            raise TypeError(
                "Expected runs-on attribute to be one of the predefined values"
            )
    return entry


def expand_parallel_entry(entry, expanded_matrix):
    """
    Expand entry with parallel-groups into multiple entries with group_id.

    Args:
        entry: Dictionary representing a test matrix entry
        expanded_matrix: List to append the expanded entries to
    """
    if "parallel-groups" in entry:
        parallel_groups = entry["parallel-groups"]

        for group_id in range(1, parallel_groups + 1):
            new_entry = entry.copy()
            new_entry["group-id"] = group_id
            expanded_matrix.append(new_entry)
    else:
        expanded_matrix.append(entry)


def read_preset_test_entries(file_path: str):
    """
    Read JSON preset file and return only test entries (excluding metadata).
    """
    with open(file_path, "r") as f:
        matrix = json.load(f)

    if not isinstance(matrix, list):
        raise ValueError("Expected JSON file to contain an array at the root level")

    return [entry for entry in matrix if "_metadata" not in entry]


def map_shared_runners_field(entry):
    """
    Maps shared_runner field from string represenation to boolean representation
    """
    shared_runners = entry.get("shared-runners")

    if shared_runners is None:
        return

    if isinstance(shared_runners, str) and shared_runners == "true":
        entry["shared-runners"] = True
    if isinstance(shared_runners, str) and shared_runners == "false":
        entry["shared-runners"] = False

    if not isinstance(entry.get("shared-runners"), bool):
        raise TypeError(
            "Expected the shared-runners field to be a boolean or a string representation of a boolean"
        )


def process_test_matrix_list(matrix_file_paths: list[str]):
    """
    Process a list of test matrices by expanding parallel groups and mapping shared runners.

    For each entry with 'parallel-groups': n, creates n total entries
    (including the original) where each has a 'group-id' starting from 1.

    Also handles 'shared-runners': true by mapping runs-on values to their
    corresponding shared runner equivalents.
    """

    test_entry_list = []
    for matrix_file_path in matrix_file_paths:
        test_entry_list.extend(read_preset_test_entries(matrix_file_path))

    expanded_matrix = []

    for entry in test_entry_list:
        map_shared_runners_field(entry)

        map_runner_name(entry)

        expand_parallel_entry(entry, expanded_matrix)

    return expanded_matrix


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python generate_test_matrix.py <file.json[:file2.json:...]>",
            file=sys.stderr,
        )
        sys.exit(1)

    matrix_file_paths = sys.argv[1]

    matrix_file_path_list = matrix_file_paths.split(":")
    for i, path in enumerate(matrix_file_path_list):
        if not Path(path).exists():
            print(
                f"Error: File '{path}' not found ({i} of {len(matrix_file_path_list)})",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        expanded_matrix = process_test_matrix_list(matrix_file_path_list)

        print(json.dumps(expanded_matrix, indent=2))

    except json.JSONDecodeError as e:
        print(
            f"Error: Invalid JSON in one of '{matrix_file_path_list}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
