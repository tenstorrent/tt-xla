#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Extract failures from base-coverage junit XML artifacts.

Walks the input directory (downloaded from call-test.yml's per-leg
`test-reports-*` artifacts) and produces:

  --out-context        Human-readable failure summary fed to Claude:
                       one section per failed test, with the message and
                       a truncated traceback.

  --out-failed-tests   Flat list of pytest node IDs that failed (one per
                       line). Used by the next iteration's matrix as the
                       `-k` narrower, and as the count source for
                       `has_failures` in the workflow.

A test that fails on multiple matrix legs (e.g. n150 and p150) is
deduped by node ID — we keep the first failure message we see.
Skipped and passing tests are ignored.

Usage:
  extract-failures.py --inputs-dir <dir> --out-context <file>
                       --out-failed-tests <file> [--max-tb-lines N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

DEFAULT_MAX_TB_LINES = 40


def parse_one(xml_path: Path) -> list[dict]:
    """Return list of failure dicts for a single junit XML file."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"::warning::could not parse {xml_path}: {e}", file=sys.stderr)
        return []
    root = tree.getroot()
    out = []
    # Top-level may be <testsuites> wrapping <testsuite>, or a single
    # <testsuite> at root. Treat both the same way.
    suites = root.findall(".//testsuite") if root.tag == "testsuites" else [root]
    for suite in suites:
        for case in suite.findall("testcase"):
            failure = case.find("failure")
            error = case.find("error")
            problem = failure if failure is not None else error
            if problem is None:
                continue
            file_attr = case.get("file") or ""
            name_attr = case.get("name") or ""
            if not (file_attr and name_attr):
                continue
            node_id = f"{file_attr}::{name_attr}"
            out.append(
                {
                    "node_id": node_id,
                    "kind": "failure" if failure is not None else "error",
                    "message": problem.get("message", "") or "",
                    "body": (problem.text or "").strip(),
                    "source": xml_path.name,
                }
            )
    return out


def dedupe(failures: list[dict]) -> list[dict]:
    """Keep first occurrence of each node_id."""
    seen: dict[str, dict] = {}
    for f in failures:
        if f["node_id"] not in seen:
            seen[f["node_id"]] = f
    return list(seen.values())


def truncate_tb(body: str, max_lines: int) -> str:
    """Keep the last `max_lines` of the traceback — that's where the
    actual error lives. Pytest pads the top with collection context."""
    lines = body.splitlines()
    if len(lines) <= max_lines:
        return body
    head = lines[: max_lines // 4]
    tail = lines[-(max_lines - len(head) - 1) :]
    return "\n".join(head + ["    ... [truncated] ..."] + tail)


def render_context(failures: list[dict], max_tb_lines: int) -> str:
    out = [f"Failing tests: {len(failures)}", ""]
    for i, f in enumerate(failures, 1):
        out.append("=" * 78)
        out.append(f"[{i}/{len(failures)}] {f['node_id']}")
        out.append(f"  kind:    {f['kind']}")
        out.append(f"  message: {f['message']}")
        out.append(f"  source:  {f['source']}")
        out.append("=" * 78)
        if f["body"]:
            out.append(truncate_tb(f["body"], max_tb_lines))
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs-dir",
        required=True,
        help="Directory containing test-reports-* subdirectories with junit XML",
    )
    ap.add_argument(
        "--out-context",
        required=True,
        help="File to write the human-readable failure summary",
    )
    ap.add_argument(
        "--out-failed-tests",
        required=True,
        help="File to write the flat list of failed node IDs",
    )
    ap.add_argument(
        "--max-tb-lines",
        type=int,
        default=DEFAULT_MAX_TB_LINES,
        help=f"Max traceback lines per failure (default {DEFAULT_MAX_TB_LINES})",
    )
    args = ap.parse_args()

    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        print(f"::error::inputs dir not found: {inputs_dir}", file=sys.stderr)
        return 2

    xml_paths = sorted(inputs_dir.rglob("*.xml"))
    if not xml_paths:
        print(f"::warning::no junit XML files under {inputs_dir}", file=sys.stderr)

    all_failures = []
    for p in xml_paths:
        all_failures.extend(parse_one(p))

    failures = dedupe(all_failures)
    failures.sort(key=lambda f: f["node_id"])

    Path(args.out_context).write_text(render_context(failures, args.max_tb_lines))
    Path(args.out_failed_tests).write_text(
        "\n".join(f["node_id"] for f in failures) + ("\n" if failures else "")
    )

    print(f"Parsed {len(xml_paths)} junit files; {len(failures)} unique failing tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
