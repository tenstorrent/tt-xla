#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Filter a `/collect-failures` failures JSON down to the categories the nightly
auto-bisect skill targets, dropping everything else.

Scope is EXCLUDE-based: keep every failing pytest test EXCEPT the two ignored areas.

OUT of scope (the ONLY things dropped)
  - vLLM / integration:      tests/integrations/**
  - performance benchmarks:  tests/benchmark/**

IN scope (everything else), labelled for the report:
  - model         : model tests — runner test_all_models_torch|_jax|test_llms_torch
                    and hand-written tests/**/models/**
  - op_by_op      : runner test_all_models_op_by_op[...]
  - op            : op tests (.../ops/...)
  - graph         : graph tests (.../graphs/...)
  - example       : tests/examples/**
  - other         : any other in-scope test (training, moe, quality, filecheck, ...)

Note: PJRT C++ unit tests (ctest -R PJRT) are gtest, not present in these pytest
logs, so they never appear here — a separate collection path would be needed.

The output preserves the input schema so it can be fed straight to
`/find-regression-boundaries`. Each kept entry in `failed_tests` gains a
`"category"` field. Dropped entries are summarised under `out_of_scope_tests`.
Timeout jobs are left in `timed_out_jobs` (the wrapper skill reports them as
UNBISECTABLE:timeout — they have no per-test result to bisect).

Usage:
  python3 .claude/scripts/filter_categories.py bisection/run_<id>_failures.json [-o OUT] [--json]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Classification. test_id from CI logs carries no pytest-marker info, so this is
# purely path/nodeid based. Scope is exclude-based: only vLLM and perf are dropped.
# ---------------------------------------------------------------------------
RUNNER_MODEL_FUNCS = ("test_all_models_torch", "test_all_models_jax", "test_llms_torch")
EXCLUDE_SUBSTRINGS = ("tests/integrations/", "tests/benchmark/")


def classify(test_id: str):
    """Return an in-scope category label, or None if the test is out of scope.

    Out of scope == vLLM/integration or perf benchmarks (only). Everything else
    is in scope; the label is descriptive for the report.
    """
    tid = (test_id or "").strip()
    if not tid:
        return None

    # The ONLY excludes: vLLM/integration and perf benchmarks.
    if any(sub in tid for sub in EXCLUDE_SUBSTRINGS):
        return None

    path = tid.split("::", 1)[0]

    if tid.startswith("tests/examples/"):
        return "example"

    if tid.startswith("tests/runner/test_models.py::"):
        func = tid.split("::", 1)[1].split("[", 1)[0]
        if func in RUNNER_MODEL_FUNCS:
            return "model"
        if func == "test_all_models_op_by_op":
            return "op_by_op"
        return "other"  # placeholder / future runner entrypoints

    if "/models/" in path:
        return "model"
    if "/ops/" in path:
        return "op"
    if "/graphs/" in path:
        return "graph"
    return "other"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "failures_json", help="Path to run_<id>_failures.json from /collect-failures"
    )
    ap.add_argument(
        "-o", "--output", help="Output path (default: <input stem>_filtered.json)"
    )
    ap.add_argument(
        "--json", action="store_true", help="Print the summary as JSON to stdout"
    )
    args = ap.parse_args(argv)

    in_path = Path(args.failures_json)
    if not in_path.is_file():
        print(f"ERROR: file not found: {in_path}", file=sys.stderr)
        return 2

    data = json.loads(in_path.read_text())
    failed = data.get("failed_tests", []) or []

    kept, dropped = [], []
    counts = Counter()
    for entry in failed:
        cat = classify(entry.get("test_id", ""))
        if cat is None:
            dropped.append(entry)
        else:
            e = dict(entry)
            e["category"] = cat
            kept.append(e)
            counts[cat] += 1

    out = dict(data)
    out["failed_tests"] = kept
    out["category_filter"] = {
        "kept": len(kept),
        "dropped": len(dropped),
        "counts": dict(counts),
        "scope": "all pytest tests except vLLM/integration and perf benchmarks",
    }
    # Keep a compact record of what was dropped, for transparency.
    out["out_of_scope_tests"] = [
        {"test_id": e.get("test_id"), "machine_type": e.get("machine_type")}
        for e in dropped
    ]

    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_name(in_path.stem + "_filtered.json")
    )
    out_path.write_text(json.dumps(out, indent=2))

    n_timeout = len(data.get("timed_out_jobs", []) or [])
    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "in_scope": len(kept),
        "out_of_scope": len(dropped),
        "by_category": dict(counts),
        "timed_out_jobs": n_timeout,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
        print(
            f"Category filter: {len(kept)} in-scope, {len(dropped)} dropped ({breakdown})"
        )
        print("  (dropped = vLLM/integration + perf benchmarks only)")
        if n_timeout:
            print(
                f"  {n_timeout} timed_out job(s) left for UNBISECTABLE:timeout handling"
            )
        print(f"  -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
