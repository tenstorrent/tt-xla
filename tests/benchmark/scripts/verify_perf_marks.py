# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Verify perf-benchmark marker selection.

Perf CI selects benchmarks by pytest markers ("<hardware> and <lane>"), one CI job per
(hardware, lane) group defined in ``.github/workflows/perf-bench-matrix.json``. This script
guards that mapping:

  * every job's mark selects a non-empty set of tests,
  * the shared driver functions (test_llm, test_vision, ...) are never selected,
  * (migration check) each job's selection equals the frozen baseline in
    ``perf_mark_map.json`` when that file is present.

It extracts applied markers statically via ``ast`` so it runs anywhere (no torch/jax/pytest
needed). With ``--pytest`` it additionally cross-checks against a live ``pytest --collect-only``
when the runtime is available.

Usage:
    python tests/benchmark/scripts/verify_perf_marks.py [--baseline perf_mark_map.json] [--pytest]
"""
import argparse
import ast
import glob
import json
import os
import subprocess
import sys

HW = ["n150", "p150", "n300_llmbox", "galaxy_wh_6u", "qb2_blackhole"]
LANES = ["nightly", "push", "nightly_accuracy", "push_accuracy"]
MARKS = set(HW) | set(LANES)
# Helpers named test_* but called directly (never run as tests) — must never carry a mark.
DRIVERS = {"test_encoder", "test_imagegen", "test_llm", "test_llm_tp", "test_vision"}

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH_DIR = os.path.join(REPO, "tests", "benchmark")
MATRIX = os.path.join(REPO, ".github", "workflows", "perf-bench-matrix.json")


def _last_attr(call):
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


def _mark_name(node):
    """pytest.mark.<X> -> 'X', else None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    ):
        return node.attr
    return None


def extract_applied():
    """Return {nodeid: set(marks in MARKS)} for every collected benchmark test/param."""
    applied = {}
    for path in sorted(glob.glob(os.path.join(BENCH_DIR, "test_*.py"))):
        rel = os.path.relpath(path, REPO)
        tree = ast.parse(open(path).read())
        param_lists = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(
                node.value, (ast.List, ast.Tuple)
            ):
                calls = [
                    e
                    for e in node.value.elts
                    if isinstance(e, ast.Call) and _last_attr(e) == "param"
                ]
                for t in node.targets:
                    if isinstance(t, ast.Name) and calls:
                        param_lists[t.id] = calls
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fmarks, pref = set(), None
            for d in node.decorator_list:
                nm = _mark_name(d)
                if nm in MARKS:
                    fmarks.add(nm)
                if (
                    isinstance(d, ast.Call)
                    and _last_attr(d) == "parametrize"
                    and len(d.args) >= 2
                    and isinstance(d.args[1], ast.Name)
                ):
                    pref = d.args[1].id
            if pref:
                for call in param_lists.get(pref, []):
                    pid, pmarks = None, set()
                    for kw in call.keywords:
                        if kw.arg == "id" and isinstance(kw.value, ast.Constant):
                            pid = kw.value.value
                        if kw.arg == "marks" and isinstance(
                            kw.value, (ast.List, ast.Tuple)
                        ):
                            pmarks = {
                                _mark_name(e)
                                for e in kw.value.elts
                                if _mark_name(e) in MARKS
                            }
                    if pid is not None:
                        applied[f"{rel}::{node.name}[{pid}]"] = fmarks | pmarks
            else:
                applied[f"{rel}::{node.name}"] = fmarks
    return applied


def select(applied, mark_expr):
    """Emulate `pytest -m "<a> and <b>"`: keep items whose marks include all tokens."""
    tokens = [t.strip() for t in mark_expr.split(" and ")]
    return {nid for nid, mk in applied.items() if all(t in mk for t in tokens)}


def pytest_collect(mark_expr):
    """Live cross-check via pytest --collect-only. Returns set(nodeid) or None if unavailable."""
    try:
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                BENCH_DIR,
                "--collect-only",
                "-q",
                "-m",
                mark_expr,
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
    except Exception:
        return None
    if "No module named pytest" in out.stderr:
        return None
    ids = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("tests/benchmark/") and "::" in line:
            ids.add(line)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        default=os.path.join(REPO, "perf_mark_map.json"),
        help="Frozen expected selection (perf_mark_map.json). Skipped if missing.",
    )
    ap.add_argument(
        "--pytest", action="store_true", help="Also cross-check via live pytest collect"
    )
    args = ap.parse_args()

    applied = extract_applied()
    jobs = json.load(open(MATRIX))
    baseline = json.load(open(args.baseline)) if os.path.exists(args.baseline) else None

    fails = 0

    # (1) every driver is unmarked
    for nid, mk in applied.items():
        if nid.split("::")[1].split("[")[0] in DRIVERS and mk:
            print(f"FAIL driver carries marks: {nid} -> {sorted(mk)}")
            fails += 1

    print(f"Jobs in matrix: {len(jobs)}")
    for job in jobs:
        mark = job["mark"]
        sel = select(applied, mark)
        # (2) non-empty
        if not sel:
            print(f"FAIL empty selection for job '{mark}'")
            fails += 1
            continue
        # (3) baseline equivalence
        note = ""
        if baseline is not None:
            hw, lane = [t.strip() for t in mark.split(" and ")]
            exp = {
                nid
                for nid, s in baseline.items()
                if hw in s["hw"] and lane in s["lanes"]
            }
            if sel != exp:
                print(
                    f"FAIL baseline mismatch '{mark}': +{sorted(sel - exp)} -{sorted(exp - sel)}"
                )
                fails += 1
                continue
            note = " == baseline"
        # (4) optional live pytest cross-check
        if args.pytest:
            live = pytest_collect(mark)
            if live is None:
                note += " (pytest unavailable)"
            elif live != sel:
                print(
                    f"FAIL pytest collect mismatch '{mark}': +{sorted(live - sel)} -{sorted(sel - live)}"
                )
                fails += 1
                continue
            else:
                note += " == pytest"
        print(f"  OK {mark:34s} {len(sel):3d} tests{note}")

    if baseline is None:
        print(
            "NOTE: baseline perf_mark_map.json not found — skipped migration equivalence check."
        )

    print("\nRESULT:", "ALL CHECKS PASSED" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
