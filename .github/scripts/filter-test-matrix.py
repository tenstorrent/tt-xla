# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Select perf-benchmark jobs from perf-bench-matrix.json by lane.

Test selection now lives in the tests themselves via pytest markers (hardware x lane),
so this script only picks which (hardware, lane) jobs to run and resolves their runner
labels. Each selected job carries a ``mark`` that call-perf-test.yml passes to
``pytest -m``.
"""
import argparse
import json
import sys


def update_runners(matrix, sh_runner):
    """Resolve each job's final ``runs-on`` label and shared-runner flag."""
    no_shared_runner = ("galaxy-wh-6u", "qb2-blackhole")
    civ2_name_map = {"n150-perf": "n150", "p150-perf": "p150b"}

    for item in matrix:
        runs_on = item.get("runs-on")
        item["runs-on-original"] = runs_on
        item["shared-runners"] = sh_runner and runs_on not in no_shared_runner
        if item["shared-runners"]:
            item["runs-on"] = (
                f"tt-ubuntu-2204-{civ2_name_map.get(runs_on, runs_on)}-stable"
            )


def main():
    parser = argparse.ArgumentParser(description="Select perf benchmark jobs by lane")
    parser.add_argument("matrix_file", help="Path to perf-bench-matrix.json")
    parser.add_argument(
        "--lane",
        required=True,
        help="Lane to run: nightly, push, nightly_accuracy, or push_accuracy",
    )
    parser.add_argument(
        "--hardware",
        action="append",
        default=[],
        help="Restrict to these runs-on labels (repeatable). Default: all in the lane.",
    )
    parser.add_argument("--sh-runner", action="store_true", help="Use shared runners")

    args = parser.parse_args()

    try:
        with open(args.matrix_file) as f:
            jobs = json.load(f)

        selected = [
            job
            for job in jobs
            if job.get("lane") == args.lane
            and (not args.hardware or job.get("runs-on") in args.hardware)
        ]

        update_runners(selected, args.sh_runner)

        if not selected:
            hw = ",".join(args.hardware) if args.hardware else "ALL"
            print(
                f"Error: no perf jobs match lane='{args.lane}' hardware='{hw}'",
                file=sys.stderr,
            )
            sys.exit(1)

        print(json.dumps({"matrix": selected}))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
