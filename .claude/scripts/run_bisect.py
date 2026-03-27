#!/usr/bin/env python3
"""
Orchestrate bisect-commit runs for all qualifying tests in a regression report.

Usage:
    python .claude/scripts/run_bisect.py <regression_report.json> <machine_type1> [machine_type2 ...]

Example:
    python .claude/scripts/run_bisect.py bisection/regression_report_23375485557-secondattempt.json n150 n300

For each test in the report that:
  - has a machine_type in the supported list
  - has both first_bad_sha and last_good_sha

A folder is created at bisection/bisect/<test_id_slug>_<machine_type>/ and a claude
instance is run (with all permissions) to bisect that test via the /bisect-commit skill.
Claude instances run sequentially — one finishes before the next starts.

Tests that do not qualify are noted in the final report with a reason.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def slugify(test_id: str) -> str:
    """Convert test_id to a filesystem-safe slug."""
    return test_id.replace("/", "_").replace(" ", "_")


def qualify_test(result: dict, supported_machine_types: list[str]) -> tuple[bool, str]:
    """
    Return (should_bisect, reason).
    reason is empty if should_bisect=True, otherwise explains why it was skipped.
    """
    machine_type = result.get("machine_type", "")
    if machine_type not in supported_machine_types:
        return False, f"Machine type '{machine_type}' not in supported list {supported_machine_types}"

    if not result.get("first_bad_sha"):
        return False, "No first_bad_sha (boundary not found or unavailable)"

    if not result.get("last_good_sha"):
        return False, "No last_good_sha (boundary not found or unavailable)"

    return True, ""


def build_claude_prompt(result: dict, log_dir: Path) -> str:
    """Build the /bisect-commit invocation string passed to claude -p."""
    test_id = result["test_id"]
    first_bad_sha = result["first_bad_sha"]
    last_good_sha = result["last_good_sha"]
    known_error = result.get("known_error", "")

    # Escape double-quotes inside known_error for safe embedding
    known_error_escaped = known_error.replace('"', '\\"')

    return (
        f'/bisect-commit '
        f'test_id="{test_id}" '
        f'first_bad_sha="{first_bad_sha}" '
        f'last_good_sha="{last_good_sha}" '
        f'known_error="{known_error_escaped}" '
        f'log_dir="{log_dir}"'
    )


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python .claude/scripts/run_bisect.py <regression_report.json> "
            "<machine_type1> [machine_type2 ...]"
        )
        sys.exit(1)

    report_arg = sys.argv[1]
    supported_machine_types = sys.argv[2:]

    # Resolve repo root
    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )

    # Resolve report path
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

    # Partition into to_bisect and skipped
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
        _save_final_report(repo_root, report_path, supported_machine_types, [], skipped)
        return

    # Ensure bisect output directory exists
    bisect_base = repo_root / "bisection" / "bisect"
    bisect_base.mkdir(parents=True, exist_ok=True)

    bisected_records: list[dict] = []

    for idx, r in enumerate(to_bisect, start=1):
        test_id = r["test_id"]
        machine_type = r["machine_type"]
        first_bad_sha = r["first_bad_sha"]
        last_good_sha = r["last_good_sha"]

        test_slug = slugify(test_id)
        folder_name = f"{test_slug}_{machine_type}"
        log_dir = bisect_base / folder_name
        log_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print(f"[{idx}/{len(to_bisect)}] Bisecting: {test_id}  [{machine_type}]")
        print(f"  bad sha:  {first_bad_sha}")
        print(f"  good sha: {last_good_sha}")
        print(f"  log dir:  {log_dir}")
        print()

        prompt = build_claude_prompt(r, log_dir)

        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "-p",
            prompt,
        ]

        start_time = datetime.now(timezone.utc)
        proc = subprocess.run(cmd, cwd=str(repo_root))
        end_time = datetime.now(timezone.utc)

        bisected_records.append(
            {
                "test_id": test_id,
                "machine_type": machine_type,
                "first_bad_sha": first_bad_sha,
                "last_good_sha": last_good_sha,
                "known_error": r.get("known_error", ""),
                "log_dir": str(log_dir),
                "exit_code": proc.returncode,
                "started_at": start_time.isoformat(),
                "finished_at": end_time.isoformat(),
            }
        )

        status = "OK" if proc.returncode == 0 else f"EXITED {proc.returncode}"
        print(f"\n  Claude finished [{status}]\n")

    _save_final_report(
        repo_root, report_path, supported_machine_types, bisected_records, skipped
    )


def _save_final_report(
    repo_root: Path,
    report_path: Path,
    supported_machine_types: list[str],
    bisected: list[dict],
    skipped: list[dict],
) -> None:
    stem = report_path.stem  # e.g. regression_report_23375485557-secondattempt
    out_path = repo_root / "bisection" / f"bisect_report_{stem}.json"

    payload = {
        "source_report": str(report_path),
        "supported_machine_types": supported_machine_types,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_bisected": len(bisected),
        "total_skipped": len(skipped),
        "bisected": bisected,
        "skipped": skipped,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nFinal report saved to: {out_path}")
    if bisected:
        print("\nBisected tests:")
        for b in bisected:
            status = "OK" if b["exit_code"] == 0 else f"EXITED {b['exit_code']}"
            print(f"  [{b['machine_type']}] {b['test_id']}  -> {status}")
            print(f"          logs: {b['log_dir']}")


if __name__ == "__main__":
    main()
