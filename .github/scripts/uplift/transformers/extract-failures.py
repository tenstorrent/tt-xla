# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Extract failure context from JUnit XML reports + pytest logs.

Produces two outputs for the transformers uplift pipeline:

  --out-context        Human-readable failure summary fed to Claude:
                       per-test sections (test id, type, message, truncated
                       traceback) followed by regex'd error excerpts from
                       any pytest.log files found in the inputs dir.

  --out-failed-tests   Flat list of pytest node IDs that failed, one per
                       line. Used by the next iteration's matrix as the
                       `-k` narrower, and as the count source for
                       `has_failures` in the workflow.

Test IDs are derived from junit's `classname` attribute (always present)
rather than `file` (sometimes absent with pytest --forked). A test that
fails on multiple matrix legs (n150 + p150) is deduped by node id — we
keep the first failure record seen.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_TB_BYTES = 2_000  # per-failure traceback cap (bounds runaway tracebacks)

LOG_ERROR_PATTERNS = re.compile(
    r"(FAILED|ERROR|ModuleNotFoundError|ImportError|AttributeError"
    r"|NameError|TypeError.*argument|cannot import name)",
    re.IGNORECASE,
)


def extract_failures_from_xml(xml_dir: Path) -> list[dict]:
    """Walk *.xml under xml_dir; return one record per failing testcase.

    Handles both <testsuites><testsuite>...</testsuite></testsuites> and
    a single root-<testsuite> shape. Test IDs are built from `classname`
    so we don't depend on the optional `file=` attribute.
    """
    failures: list[dict] = []
    for xml_file in sorted(xml_dir.rglob("*.xml")):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as e:
            print(f"::warning::could not parse {xml_file}: {e}", file=sys.stderr)
            continue
        root = tree.getroot()
        suites = root.findall(".//testsuite") if root.tag == "testsuites" else [root]
        for suite in suites:
            for case in suite.findall("testcase"):
                failure = case.find("failure")
                error = case.find("error")
                # Use `is not None` — Element truth-value is deprecated
                # and an element with no children evaluates as falsy.
                problem = failure if failure is not None else error
                if problem is None:
                    continue
                classname = case.get("classname", "") or ""
                name = case.get("name", "") or ""
                if not name:
                    continue
                # tests.runner.test_models -> tests/runner/test_models.py
                node_id = (
                    f"{classname.replace('.', '/')}.py::{name}" if classname else name
                )
                body = problem.text or ""
                failures.append(
                    {
                        "test": node_id,
                        "type": problem.tag,
                        "message": (problem.get("message") or "").strip(),
                        "traceback": body[:MAX_TB_BYTES],
                        "source": xml_file.name,
                    }
                )
    return failures


def dedupe_by_test(failures: list[dict]) -> list[dict]:
    """Keep first occurrence of each node id (same test can fail on
    multiple matrix legs and produce one record per leg)."""
    seen: dict[str, dict] = {}
    for f in failures:
        seen.setdefault(f["test"], f)
    return sorted(seen.values(), key=lambda f: f["test"])


def extract_errors_from_logs(logs_dir: Path) -> list[dict]:
    """Grep every pytest.log under logs_dir for failure/error patterns.
    Around each match, capture -2 / +10 lines of context. Deduplicate
    overlapping chunks; cap at 30 unique chunks per log file."""
    excerpts: list[dict] = []
    for log_file in sorted(logs_dir.rglob("pytest.log")):
        try:
            lines = log_file.read_text(errors="replace").splitlines(keepends=True)
        except OSError:
            continue
        chunks = []
        for i, line in enumerate(lines):
            if LOG_ERROR_PATTERNS.search(line):
                start = max(0, i - 2)
                end = min(len(lines), i + 11)
                chunks.append("".join(lines[start:end]))
        if not chunks:
            continue
        # Dedup overlapping chunks; key by first 200 chars.
        seen, unique = set(), []
        for c in chunks:
            k = c[:200]
            if k in seen:
                continue
            seen.add(k)
            unique.append(c)
        excerpts.append(
            {
                "log_file": log_file.parent.name,
                "errors": unique[:30],
            }
        )
    return excerpts


def format_output(failures: list[dict], log_excerpts: list[dict]) -> str:
    parts: list[str] = []
    if failures:
        parts.append(f"=== FAILED TESTS ({len(failures)} total) ===\n")
        for f in failures:
            parts.append(f"TEST: {f['test']}")
            parts.append(f"TYPE: {f['type']}")
            if f["message"]:
                parts.append(f"MESSAGE: {f['message']}")
            if f["traceback"]:
                parts.append(f"TRACEBACK:\n{f['traceback']}")
            parts.append("---")
    if log_excerpts:
        parts.append("\n=== ERROR EXCERPTS FROM LOGS ===\n")
        for ex in log_excerpts:
            parts.append(f"--- Log: {ex['log_file']} ---")
            for err in ex["errors"]:
                parts.append(err)
                parts.append("~~~")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs-dir",
        required=True,
        help="Directory containing test-reports-*/ subdirs (junit XML) and "
        "optionally test-log-*/ subdirs (pytest.log files).",
    )
    ap.add_argument(
        "--out-context",
        required=True,
        help="File to write the human-readable failure summary.",
    )
    ap.add_argument(
        "--out-failed-tests",
        required=True,
        help="File to write the flat list of failed node IDs.",
    )
    args = ap.parse_args()

    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        print(f"::error::inputs dir not found: {inputs_dir}", file=sys.stderr)
        return 2

    xml_files = sorted(inputs_dir.rglob("*.xml"))
    if not xml_files:
        print(f"::warning::no junit XML files under {inputs_dir}", file=sys.stderr)

    raw_failures = extract_failures_from_xml(inputs_dir)
    failures = dedupe_by_test(raw_failures)
    log_excerpts = extract_errors_from_logs(inputs_dir)

    Path(args.out_context).write_text(format_output(failures, log_excerpts))
    Path(args.out_failed_tests).write_text(
        "\n".join(f["test"] for f in failures) + ("\n" if failures else "")
    )

    print(
        f"Parsed {len(xml_files)} junit files; "
        f"{len(failures)} unique failing tests; "
        f"{sum(len(e['errors']) for e in log_excerpts)} log excerpts."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
