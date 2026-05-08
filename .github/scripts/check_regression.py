# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys

REGRESSION_THRESHOLD_PCT = 5.0


def check_regression(label, prev, current):
    if prev is None or current is None:
        print(f"{label}: missing data, skipping")
        return False
    drop_pct = (prev - current) / prev * 100
    if drop_pct >= REGRESSION_THRESHOLD_PCT:
        print(
            f"{label} regression > {REGRESSION_THRESHOLD_PCT}% detected! Dropped by {drop_pct:.4f}%"
        )
        return True
    print(f"{label} check passed. Diff: {drop_pct:.4f}%")
    return False


def check_accuracy(prev_report, current_report):
    prev = {}
    for e in prev_report:
        prev[e["NAME"]] = float(e["LAST_VALUE"])
    current = {}
    for m in current_report.get("measurements", []):
        current[m["measurement_name"]] = float(m["value"])

    print(
        f"Previous: top1={prev.get('top1_accuracy')}  top5={prev.get('top5_accuracy')}"
    )
    print(
        f"Current:  top1={current.get('top1_accuracy')}  top5={current.get('top5_accuracy')}"
    )

    failed = check_regression(
        "Top-1 accuracy", prev.get("top1_accuracy"), current.get("top1_accuracy")
    )
    failed |= check_regression(
        "Top-5 accuracy", prev.get("top5_accuracy"), current.get("top5_accuracy")
    )
    return failed


def check_perf(prev_report, current_report):
    prev = {}
    for e in prev_report:
        prev[e["NAME"]] = float(e["LAST_VALUE"])
    current = {}
    for m in current_report.get("measurements", []):
        current[m["measurement_name"]] = float(m["value"])

    if "total_samples" not in prev or "total_time" not in prev:
        print("Previous report missing total_samples/total_time, skipping")
        return False
    if "total_samples" not in current or "total_time" not in current:
        print("Current report missing total_samples/total_time, skipping")
        return False

    prev_sps = prev["total_samples"] / prev["total_time"]
    current_sps = current["total_samples"] / current["total_time"]

    print(
        f"Previous: samples={prev['total_samples']}  time={prev['total_time']}  sps={prev_sps:.6f}"
    )
    print(
        f"Current:  samples={current['total_samples']}  time={current['total_time']}  sps={current_sps:.6f}"
    )

    return check_regression("Samples/sec", prev_sps, current_sps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prev-report", required=True, help="Previous run results as a JSON string"
    )
    parser.add_argument(
        "--current-report", required=True, help="Path to current perf report JSON file"
    )
    parser.add_argument("--mode", choices=["perf", "accuracy"], default="perf")
    args = parser.parse_args()

    prev_report = json.loads(args.prev_report)
    if prev_report == []:
        print("No previous report found, skipping regression check")
        sys.exit(0)

    with open(args.current_report) as f:
        current_report = json.load(f)

    if args.mode == "accuracy":
        failed = check_accuracy(prev_report, current_report)
    else:
        failed = check_perf(prev_report, current_report)

    if failed:
        exit(1)


if __name__ == "__main__":
    main()
