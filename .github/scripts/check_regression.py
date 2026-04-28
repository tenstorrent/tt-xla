# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys

REGRESSION_THRESHOLD_PCT = 5.0


def lookup_prev(report, name):
    for entry in report:
        if entry.get("NAME") == name:
            return float(entry["LAST_VALUE"])
    return None


def lookup_current(report, name):
    for m in report.get("measurements", []):
        if m.get("measurement_name") == name:
            return float(m["value"])
    return None


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
    prev_top1 = lookup_prev(prev_report, "top1_accuracy")
    prev_top5 = lookup_prev(prev_report, "top5_accuracy")
    current_top1 = lookup_current(current_report, "top1_accuracy")
    current_top5 = lookup_current(current_report, "top5_accuracy")

    print(f"Previous: top1={prev_top1}  top5={prev_top5}")
    print(f"Current:  top1={current_top1}  top5={current_top5}")

    failed = check_regression("Top-1 accuracy", prev_top1, current_top1)
    failed |= check_regression("Top-5 accuracy", prev_top5, current_top5)
    return failed


def check_perf(prev_report, current_report):
    prev_samples = lookup_prev(prev_report, "total_samples")
    prev_time = lookup_prev(prev_report, "total_time")
    current_samples = lookup_current(current_report, "total_samples")
    current_time = lookup_current(current_report, "total_time")

    if prev_samples is None or prev_time is None:
        print("Previous report missing total_samples/total_time, skipping")
        return False
    if current_samples is None or current_time is None:
        print("Current report missing total_samples/total_time, skipping")
        return False

    prev_sps = prev_samples / prev_time
    current_sps = current_samples / current_time

    print(f"Previous: samples={prev_samples}  time={prev_time}  sps={prev_sps:.6f}")
    print(
        f"Current:  samples={current_samples}  time={current_time}  sps={current_sps:.6f}"
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
    parser.add_argument("--accuracy-testing", action="store_true")
    args = parser.parse_args()

    prev_report = json.loads(args.prev_report)
    if prev_report == []:
        print("No previous report found, skipping regression check")
        sys.exit(0)

    with open(args.current_report) as f:
        current_report = json.load(f)

    if args.accuracy_testing:
        failed = check_accuracy(prev_report, current_report)
    else:
        failed = check_perf(prev_report, current_report)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
