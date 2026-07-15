#!/usr/bin/env python3
"""Write a Markdown table of in-scope nightly failures (test name + arch).

Run this every time right after fetching/filtering the failure results (Phase 2
of /bisect-nightly). It reads the filtered failures JSON, enriches the `reason`
column from the persistent ledger (bisection/bisected.json) when an entry exists,
and writes bisection/nightly_<run_id>_failures_table.md.

Usage:
  python3 .claude/scripts/write_failures_table.py bisection/run_<run_id>_failures_filtered.json
    [--ledger bisection/bisected.json] [--out <path>]

The output format matches bisection/nightly_29666632980_failures_table.md:
  | # | test | arch | reason | xfail? |
plus a "Hard cases" section for timed-out jobs.
"""
import argparse
import json
import os
import sys

ARCH_ORDER = {
    "n150": 0, "p150": 1, "n300": 2, "n300-llmbox": 3, "llmbox": 3,
    "galaxy-wh-6u": 4, "galaxy-bh": 5, "qb2-blackhole": 6,
}


def short_name(test_id):
    """Node id minus the file path prefix: everything after the last '::'."""
    return test_id.split("::")[-1] if "::" in test_id else test_id


def load_ledger(path):
    if not path or not os.path.exists(path):
        return {}
    data = json.load(open(path))
    return data.get("entries", data)


def reason_from_ledger(entry, raw_error):
    """Build the human reason string. Mirrors the reference table wording."""
    if not entry:
        err = (raw_error or "").strip().splitlines()[0] if raw_error else ""
        return (f"UNCLASSIFIED; {err}" if err else "UNCLASSIFIED"), ""
    status = entry.get("status", "")
    since = entry.get("first_bad_date") or entry.get("first_bad_run_id") or ""
    since_txt = f"; since {since}" if since else ""
    if status == "bisected":
        pr = entry.get("blame_pr", "")
        sha = entry.get("blame_sha", "")
        sha_txt = f" (`{sha[:8]}`)" if sha else ""
        return f"BISECTED regression -> {pr}{sha_txt}{since_txt}", "xfail-safe"
    if status == "pending":
        return f"EXISTING deterministic failure, bisection pending{since_txt}", "xfail-safe"
    if status == "checkpoint_drift":
        return f"checkpoint drift (HF), not a commit regression{since_txt}", ""
    if status == "group_only":
        return f"group-only failure (passes standalone){since_txt}", ""
    if status == "timeout":
        return f"timeout{since_txt}", ""
    err = (entry.get("known_error") or raw_error or "").strip()
    return (f"{status or 'tracked'}{since_txt}; {err}".strip("; "), "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filtered_json", help="bisection/run_<run_id>_failures_filtered.json")
    ap.add_argument("--ledger", default=None, help="path to bisected.json (auto-detected next to the json)")
    ap.add_argument("--out", default=None, help="output md path (default nightly_<run_id>_failures_table.md)")
    args = ap.parse_args()

    d = json.load(open(args.filtered_json))
    run_id = d.get("run_id")
    run_date = d.get("run_date", "")
    sha = d.get("sha", "")
    workflow = d.get("workflow_name", "On nightly")
    failed = d.get("failed_tests", [])
    timed_out = d.get("timed_out_jobs", [])

    bis_dir = os.path.dirname(os.path.abspath(args.filtered_json))
    ledger_path = args.ledger or os.path.join(bis_dir, "bisected.json")
    ledger = load_ledger(ledger_path)

    # sort: test_id A->Z, then arch in fixed order
    failed = sorted(failed, key=lambda t: (t["test_id"], ARCH_ORDER.get(t.get("machine_type"), 9)))

    from collections import Counter
    cats = Counter(t.get("category", "other") for t in failed)
    cat_txt = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))

    lines = []
    lines.append(f"# Nightly failures — {workflow} run {run_id} ({run_date[:10]}, sha `{sha[:10]}`)")
    lines.append("")
    lines.append(
        f"{len(failed)} in-scope failed tests ({cat_txt}; vLLM/perf excluded) "
        f"+ {len(timed_out)} job-level hard cases. Reasons from `bisection/bisected.json`."
    )
    lines.append("")
    lines.append("## In-scope failed tests")
    lines.append("")
    lines.append("| # | test | arch | reason | xfail? |")
    lines.append("|---|------|------|--------|--------|")
    for i, t in enumerate(failed, 1):
        arch = t.get("machine_type") or "?"
        key = f"{t['test_id']}||{arch}"
        entry = ledger.get(key)
        reason, xfail = reason_from_ledger(entry, t.get("raw_error"))
        lines.append(f"| {i} | `{short_name(t['test_id'])}` | {arch} | {reason} | {xfail} |")
    lines.append("")

    if timed_out:
        lines.append("## Hard cases (timed-out jobs — not bisectable)")
        lines.append("")
        lines.append("| # | job | last test executing |")
        lines.append("|---|-----|---------------------|")
        for i, j in enumerate(timed_out, 1):
            job = j.get("job_name", str(j.get("job_id", "")))
            last = j.get("last_test_executing") or "?"
            lines.append(f"| {i} | {job} | `{short_name(last)}` |")
        lines.append("")

    out = args.out or os.path.join(bis_dir, f"nightly_{run_id}_failures_table.md")
    open(out, "w").write("\n".join(lines))
    print(f"Wrote {len(failed)} failures ({len(timed_out)} hard cases) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
