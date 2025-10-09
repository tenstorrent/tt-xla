# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Calculate test timeout based on collected tests and their durations.

If any tests are missing from the durations file, use a default timeout of 4 hours (240 minutes).
If all tests are found, use the estimated duration multiplied by 3 for safety margin.
"""

import json
import argparse


def parse_collected_tests(collected_output):
    """Parse collected tests from file, filtering lines containing '::'."""
    with open(collected_output, "r") as f:
        return [line.strip() for line in f if "::" in line]


def load_test_durations(durations_file):
    """Load test durations from JSON file."""
    with open(durations_file, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate test timeout based on collected tests and durations"
    )
    parser.add_argument(
        "--collected-output",
        required=True,
        help="File containing output from pytest --collect-only -q",
    )
    parser.add_argument(
        "--durations-file",
        default=".test_durations",
        help="Path to test durations file",
    )
    parser.add_argument(
        "--default-timeout", type=int, default=240, help="Default timeout in minutes"
    )

    args = parser.parse_args()

    collected_tests = parse_collected_tests(args.collected_output)
    durations = load_test_durations(args.durations_file)

    print(f"Collected {len(collected_tests)} tests")
    print(f"Loaded {len(durations)} test durations from {args.durations_file}")

    missing_tests = set(collected_tests) - set(durations.keys())

    if missing_tests:
        print(f"Found {len(missing_tests)} tests missing from durations file:")
        for test in list(missing_tests)[:10]:
            print(f"  - {test}")
        if len(missing_tests) > 10:
            print(f"  ... and {len(missing_tests) - 10} more")
        print(f"Using default timeout of {args.default_timeout} minutes")
        timeout_minutes = args.default_timeout
    else:
        total_duration_seconds = sum(durations[test] for test in collected_tests)
        timeout_minutes = max(int((total_duration_seconds / 60) * 3), 10)
        print(
            f"Total duration: {total_duration_seconds:.1f} seconds ({total_duration_seconds/60:.1f} minutes)"
        )
        print(f"Timeout with 3x safety margin: {timeout_minutes} minutes")

    print(timeout_minutes)


if __name__ == "__main__":
    main()
