#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Parse CI logs to extract every test name and its result."""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

LOGDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ci-logs")

# Regex to strip the CI log line prefix (job name + UNKNOWN STEP + timestamp)
# Example: "test / test ... (lb-blackhole, ...)	UNKNOWN STEP	2026-03-29T16:26:10.2774352Z PASSED"
CI_PREFIX = re.compile(r"^.*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s?")
# ANSI escape codes
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Test ID pattern that handles parameterized tests with spaces in brackets.
# e.g., tests/path/test.py::test_name[float32-(256, 128, 64)]
# Uses [^\s:]+ for the file path (no spaces/colons), then ::word segments,
# then optional [...] for parameters (which may contain spaces).
TID = r"tests/[^\s:]+(?:::\w+)+(?:\[.*?\])?"


def strip_line(line: str) -> str:
    """Remove CI log prefix and ANSI codes from a line."""
    line = CI_PREFIX.sub("", line)
    line = ANSI.sub("", line)
    return line.rstrip()


def parse_log(logfile: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse a single log file and return ({test_id: result}, {test_id: reason})."""
    raw_lines = logfile.read_text(errors="replace").splitlines()
    lines = [strip_line(l) for l in raw_lines]

    results = {}  # test_id -> PASSED/FAILED/ERROR/SKIPPED
    reasons = {}  # test_id -> failure/error reason (only for FAILED/ERROR)

    # 1. Collect test IDs from multiple sources:
    #    a) pytest-split collection listing (standalone test ID per line)
    #    b) "slowest durations" section: "179.78s call tests/...::test[...]"
    #       This is especially important for manual/truncated logs where the
    #       collection listing and verbose output may be incomplete.
    collected = []
    for line in lines:
        # Standalone test ID (collection listing)
        m = re.match(rf"^({TID})\s*$", line)
        if m:
            collected.append(m.group(1))
            continue
        # Durations section: "<time>s call|setup|teardown <test_id>"
        m = re.match(rf"^\d+\.\d+s\s+(?:call|setup|teardown)\s+({TID})\s*$", line)
        if m:
            collected.append(m.group(1))

    # 2. Extract FAILED and ERROR from the short test summary section.
    #    Format: "FAILED tests/...::test[...] - error message"
    #    Format: "ERROR tests/...::test[...] - error message"
    in_summary = False
    for line in lines:
        if "short test summary info" in line:
            in_summary = True
            continue
        if in_summary:
            m = re.match(rf"^(FAILED|ERROR)\s+({TID})(?:\s+-\s+(.+))?", line)
            if m:
                status, test_id, reason = m.group(1), m.group(2), m.group(3)
                results[test_id] = status
                if reason:
                    reasons[test_id] = reason
            # Summary section ends at the final pytest summary line
            if re.match(r"^=+\s+\d+\s+\w+", line):
                break

    # 3. Extract SKIPPED, PASSED, and XFAIL from verbose test output lines.
    #    Format: "tests/...::test[...] PASSED|SKIPPED|XFAIL"
    #    Also handle bare "PASSED", "XFAIL" on their own line (long-running tests
    #    print lots of output between test name and result). Track last seen test
    #    name to attribute bare result lines.
    last_test_id = None
    for line in lines:
        # Line with test ID and result on same line
        m = re.match(rf"^({TID})\s+(PASSED|SKIPPED|XFAIL)\s*$", line)
        if m:
            test_id, status = m.group(1), m.group(2)
            if test_id not in results:
                results[test_id] = status
            last_test_id = None
            continue

        # Line that starts a new test (verbose -sv output): "tests/...::test Running ..."
        # or just "tests/...::test <anything that's not a result>"
        m = re.match(rf"^({TID})\s", line)
        if m:
            last_test_id = m.group(1)
            continue

        # Bare result on its own line (long test output separated name from result)
        if line in ("PASSED", "XFAIL", "SKIPPED"):
            if last_test_id and last_test_id not in results:
                results[last_test_id] = line
            last_test_id = None
            continue

    # For forge model tests where PASSED/XFAIL appears on its own line and we
    # couldn't attribute it (no last_test_id), infer from collected tests.
    if collected:
        for test_id in collected:
            if test_id not in results:
                results[test_id] = "PASSED"

    # 4. Reconcile against the pytest summary line to catch tests whose names
    #    are not visible in the log (e.g., hidden in "(N durations < 0.005s hidden)").
    #    Parse: "= 11 failed, 14 passed, 1 skipped, ... in 1352.62s ... ="
    for line in lines:
        m = re.match(r"^=+\s+(.*?)\s+in\s+\d+\.\d+s", line)
        if not m:
            continue
        summary = m.group(1)
        # Map summary categories to our result keys
        # pytest summary: passed, failed, error, skipped, xfailed, xpassed
        summary_map = {
            "passed": "PASSED",
            "failed": "FAILED",
            "error": "ERROR",
            "skipped": "SKIPPED",
            "xfailed": "XFAIL",
        }
        from collections import Counter

        found_counts = Counter(results.values())
        for word, result_key in summary_map.items():
            sm = re.search(rf"(\d+) {word}", summary)
            expected = int(sm.group(1)) if sm else 0
            found = found_counts.get(result_key, 0)
            if expected > found:
                gap = expected - found
                for i in range(gap):
                    placeholder = f"UNKNOWN-{logfile.stem}-{result_key}-{i}"
                    results[placeholder] = result_key
        break  # only process last summary line (already handled by tail matching)

    return results, reasons


def main():
    all_results = {}  # test_id -> (result, job_id)
    all_reasons = {}  # test_id -> reason
    manual_jobs = set()

    for logfile in sorted(LOGDIR.glob("*.log")):
        job_id = logfile.stem

        # Detect manually-pulled logs
        first_line = logfile.read_text(errors="replace").split("\n", 1)[0]
        if "log not found" in first_line:
            manual_jobs.add(job_id)

        results, reasons = parse_log(logfile)
        for test_id, result in results.items():
            all_results[test_id] = (result, job_id)
        all_reasons.update(reasons)

    # Print all tests sorted by result then name
    result_order = {"FAILED": 0, "ERROR": 1, "SKIPPED": 2, "XFAIL": 3, "PASSED": 4}
    sorted_tests = sorted(
        all_results.items(), key=lambda x: (result_order.get(x[1][0], 9), x[0])
    )

    counts = defaultdict(int)
    counts_ui = defaultdict(int)  # excluding manual jobs

    for test_id, (result, job_id) in sorted_tests:
        manual_tag = " *" if job_id in manual_jobs else ""
        reason = all_reasons.get(test_id, "")
        reason_suffix = f"  -- {reason}" if reason else ""
        print(f"{result:8s}  {test_id}{manual_tag}{reason_suffix}")
        counts[result] += 1
        if job_id not in manual_jobs:
            counts_ui[result] += 1

    print()
    print(f"Total unique tests: {len(sorted_tests)}")
    for result in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]:
        if counts[result]:
            print(f"  {result}: {counts[result]}")

    if manual_jobs:
        print(f"\nExcluding manual logs ({', '.join(sorted(manual_jobs))}):")
        ui_total = sum(counts_ui.values())
        print(f"  Total: {ui_total}")
        # Map to GitHub UI categories
        ui_passed = counts_ui["PASSED"]
        ui_failed = counts_ui["FAILED"] + counts_ui["ERROR"]
        ui_skipped = counts_ui["SKIPPED"] + counts_ui["XFAIL"]
        print(f"  Passed: {ui_passed}, Failed: {ui_failed}, Skipped: {ui_skipped}")


if __name__ == "__main__":
    main()
