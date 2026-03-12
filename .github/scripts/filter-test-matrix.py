# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import sys
from pathlib import Path


def flatten_matrix(data):
    """Flatten the matrix."""
    matrix = []
    for proj in data:
        test_defaults = proj.get("test-defaults", {})
        for test in proj.get("tests", []):
            merged_test = {**test_defaults, **test, "project": proj["project"]}

            runs_on = merged_test.get("runs-on", [])
            if isinstance(runs_on, list):
                matrix.extend({**merged_test, "runs-on": runner} for runner in runs_on)
            else:
                matrix.append(merged_test)

    return matrix


def get_unique_runners(matrix):
    """Get unique runners from the matrix."""
    return set(item.get("runs-on") for item in matrix if item.get("runs-on"))


def filter_matrix_adv(matrix, adv_filter):
    """
    Filter matrix based on advanced filter.
    Filter is JSON array with sets of conditions.
    Each condition might have elements:
      - "runs-on": machine(s) on which the test should run. If ommited, condition will be applied to all unskipped machines. Parameter can be string or array of strings.
      - "filter": string that should be present in the test name.
      - "accuracy-testing": whether to include accuracy testing or not.
      - "skip": whether to skip tests matching the condition or not. If ommited, it is assumed to be true.
    """
    # Create initial structure with all runners marked as skip=True
    runners = get_unique_runners(matrix)
    runner_conditions = {runner: {"skip": True} for runner in runners}

    # Apply conditions from adv_filter
    for condition in adv_filter:
        condition_runners = condition.get("runs-on")

        # Determine which runners this condition applies to
        if condition_runners:
            if isinstance(condition_runners, list):
                target_runners = condition_runners
            else:
                target_runners = [condition_runners]

            # Set all target runners skip condition to false
            for runner in target_runners:
                if runner not in runner_conditions:
                    print(
                        f"Error: Runner '{runner}' specified in filter JSON does not exist in the matrix",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                runner_conditions[runner]["skip"] = False
        else:
            # If no runs-on specified, apply to all runners
            target_runners = list(runners)

        # Update conditions for target runners
        for runner in target_runners:
            if condition.get("filter") is not None:
                if "filter" not in runner_conditions[runner]:
                    runner_conditions[runner]["filter"] = []
                runner_conditions[runner]["filter"].append(condition["filter"].lower())
            if condition.get("accuracy-testing") is not None:
                runner_conditions[runner]["accuracy-testing"] = condition[
                    "accuracy-testing"
                ]
            if condition.get("skip") is not None:
                runner_conditions[runner]["skip"] = condition["skip"]

    # Filter the matrix based on the constructed runner conditions
    filtered_matrix = []
    for item in matrix:
        runner = item.get("runs-on")
        if runner in runner_conditions:
            conditions = runner_conditions[runner]
            if conditions["skip"]:
                continue
            if "filter" in conditions:
                if not any(
                    f in item.get("name", "").lower() for f in conditions["filter"]
                ):
                    continue
            if "accuracy-testing" in conditions and conditions[
                "accuracy-testing"
            ] != item.get("accuracy-testing", False):
                continue
            filtered_matrix.append(item)

    return filtered_matrix


def update_runners(matrix, sh_runner):
    """Update runner names based on shared runner flag."""
    runner_map = {"p150": "p150b"} if sh_runner else {"n150": "n150-perf"}

    for item in matrix:
        item["runs-on-original"] = item.get("runs-on")
        if item.get("runs-on") in runner_map:
            item["runs-on"] = runner_map[item["runs-on"]]

    return matrix


def main():
    parser = argparse.ArgumentParser(description="Filter benchmark matrix")
    parser.add_argument("matrix_file", help="Path to benchmark matrix JSON file")
    parser.add_argument("adv_filter", help="JSON file with advanced filter")
    parser.add_argument("--sh-runner", action="store_true", help="Use shared runners")

    args = parser.parse_args()
    matrix_skip = "false"

    try:
        with open(args.matrix_file) as f:
            data = json.load(f)

        with open(args.adv_filter) as f:
            adv_filter = json.load(f)

        matrix = flatten_matrix(data)
        filtered = filter_matrix_adv(matrix, adv_filter)

        update_runners(filtered, args.sh_runner)

        if not filtered:
            print("Error: No matching tests found", file=sys.stderr)
            matrix_skip = "true"

        result = {"matrix": filtered, "matrix_skip": matrix_skip}
        print(json.dumps(result))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
