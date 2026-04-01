#!/usr/bin/env python3
"""
Dispatch CI bisect jobs for all qualifying tests in a regression report.

Usage:
    python .claude/scripts/run_bisect.py <regression_report.json> <machine_type1> [machine_type2 ...]

Example:
    python .claude/scripts/run_bisect.py bisection/regression_report_23375485557.json n150 n300

For each test in the report that:
  - has a machine_type in the supported list
  - has both first_bad_sha and last_good_sha

A single GitHub Actions workflow dispatch is sent to workflow-bisect.yml with all
qualifying tests in parallel. The workflow handles bisection in CI, including mlir
uplift detection.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


GITHUB_REPO = "tenstorrent/tt-xla"
WORKFLOW_FILE = "workflow-bisect.yml"


def qualify_test(result: dict, supported_machine_types: list[str]) -> tuple[bool, str]:
    machine_type = result.get("machine_type", "")
    if machine_type not in supported_machine_types:
        return False, f"Machine type '{machine_type}' not in supported list {supported_machine_types}"

    if not result.get("first_bad_sha"):
        return False, "No first_bad_sha (boundary not found)"

    if not result.get("last_good_sha"):
        return False, "No last_good_sha (boundary not found)"

    return True, ""


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python .claude/scripts/run_bisect.py <regression_report.json> "
            "<machine_type1> [machine_type2 ...]"
        )
        sys.exit(1)

    report_arg = sys.argv[1]
    supported_machine_types = sys.argv[2:]

    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )

    report_path = Path(report_arg)
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    if not report_path.exists():
        print(f"ERROR: Report file not found: {report_path}")
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    results = report.get("results", [])
    if not results:
        print("No results found in report.")
        sys.exit(0)

    print(f"Loaded {len(results)} tests from: {report_path}")
    print(f"Supported machine types: {supported_machine_types}")
    print()

    to_bisect: list[dict] = []
    skipped: list[dict] = []

    for r in results:
        ok, reason = qualify_test(r, supported_machine_types)
        if ok:
            to_bisect.append(r)
        else:
            skipped.append(
                {
                    "test_id": r.get("test_id", "unknown"),
                    "machine_type": r.get("machine_type", "unknown"),
                    "reason": reason,
                }
            )

    print(f"Tests to bisect: {len(to_bisect)}")
    print(f"Tests skipped:   {len(skipped)}")
    if skipped:
        print("\nSkipped tests:")
        for s in skipped:
            print(f"  [{s['machine_type']}] {s['test_id']}")
            print(f"          Reason: {s['reason']}")
    print()

    if not to_bisect:
        print("Nothing to bisect.")
        return

    # Build the bisect_jobs JSON for the workflow
    bisect_jobs = [
        {
            "test_id": r["test_id"],
            "good_sha": r["last_good_sha"],
            "bad_sha": r["first_bad_sha"],
            "machine": r["machine_type"],
        }
        for r in to_bisect
    ]

    print("Dispatching bisect jobs to CI:")
    for j in bisect_jobs:
        print(f"  [{j['machine']}] {j['test_id']}")
        print(f"           good: {j['good_sha']}  bad: {j['bad_sha']}")
    print()

    bisect_jobs_json = json.dumps(bisect_jobs)

    cmd = [
        "gh", "workflow", "run", WORKFLOW_FILE,
        "--repo", GITHUB_REPO,
        "-f", f"bisect_jobs={bisect_jobs_json}",
    ]

    print(f"Running: gh workflow run {WORKFLOW_FILE} --repo {GITHUB_REPO} -f bisect_jobs=<{len(bisect_jobs)} jobs>")
    result = subprocess.run(cmd, cwd=str(repo_root))

    if result.returncode != 0:
        print(f"\nERROR: workflow dispatch failed (exit {result.returncode})")
        sys.exit(result.returncode)

    print(f"\nDispatched {len(to_bisect)} bisect jobs to CI in parallel.")
    print(f"Monitor at: https://github.com/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}")


if __name__ == "__main__":
    main()
