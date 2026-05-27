#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Aggregate report generator for rel_l2 JSONL output from
test_matmul_gpt_oss_120b.py / test_layer_gpt_oss_120b.py.

Produces grouped tables (by op, dtype, fidelity, layer) with mean rel_l2,
plus top movers. Designed for comparing two wheel builds before/after a
compiler change.

Each JSONL line: {"test_id": ..., "rel_l2": ..., "pcc": ...}

The test_id is parsed as:
    .../test_matmul_gpt_oss_120b.py::test_matmul_gpt_oss_120b[layer_N__op-optX_dtype_fidelity_fp32YYY]

Usage:
    python scripts/report_rel_l2.py before.jsonl after.jsonl
    python scripts/report_rel_l2.py before.jsonl after.jsonl --format markdown
    python scripts/report_rel_l2.py before.jsonl after.jsonl --top 20
    python scripts/report_rel_l2.py single_run.jsonl                  # single-run summary
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PARAM_RE = re.compile(r"\[(.+?)\]")


def load_jsonl(path: Path) -> Dict[str, Tuple[float, float]]:
    """Return {parametrize_id: (rel_l2, pcc)}."""
    out: Dict[str, Tuple[float, float]] = {}
    for line_no, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"WARNING {path}:{line_no} skipping malformed: {exc}", file=sys.stderr)
            continue
        tid = entry.get("test_id", "")
        m = _PARAM_RE.search(tid)
        if not m:
            continue
        out[m.group(1)] = (entry.get("rel_l2", float("nan")), entry.get("pcc", float("nan")))
    return out


def parse_param(p: str) -> Optional[Tuple[str, str, str, str, str, str]]:
    """Parse layer_N__op-optX_dtype_fidelity_fp32YYY into 6 fields.

    Returns None if the id does not follow the expected schema.
    """
    if "-" not in p or "__" not in p:
        return None
    layer_op, cc = p.split("-", 1)
    layer, op = layer_op.split("__", 1)
    parts = cc.split("_")
    if len(parts) < 4:
        return None
    return layer, op, parts[0], parts[1], parts[2], parts[3]


def pct(b: float, a: float) -> float:
    return (a - b) / b * 100 if b else 0.0


def aggregate(data: Dict[str, Tuple[float, float]], key_fn) -> Dict[Tuple, List[float]]:
    """Group rel_l2 values by key_fn(parsed_param)."""
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for param, (rel_l2, _) in data.items():
        parsed = parse_param(param)
        if parsed is None:
            continue
        key = key_fn(parsed)
        groups[key].append(rel_l2)
    return groups


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _row_text(cols: List[str], widths: List[int]) -> str:
    return " | ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(cols, widths)))


def emit_compare_table(
    title: str,
    rows: List[List[str]],
    headers: List[str],
    widths: List[int],
    fmt: str,
) -> None:
    print()
    if fmt == "markdown":
        print(f"### {title}")
        print()
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
    else:
        print("=" * sum(widths + [3 * (len(widths) - 1)]))
        print(title)
        print("=" * sum(widths + [3 * (len(widths) - 1)]))
        print(_row_text(headers, widths))
        print("-" * sum(widths + [3 * (len(widths) - 1)]))
        for row in rows:
            print(_row_text(row, widths))


def emit_single_table(title: str, rows: List[List[str]], headers: List[str], widths: List[int], fmt: str) -> None:
    emit_compare_table(title, rows, headers, widths, fmt)


# ---------------------------------------------------------------------------
# Compare mode (two runs)
# ---------------------------------------------------------------------------


def report_compare(before: Dict, after: Dict, top: int, fmt: str) -> None:
    common = sorted(set(before) & set(after))

    def grouped(key_fn) -> List[Tuple[Tuple, float, float, int]]:
        b_groups = defaultdict(list)
        a_groups = defaultdict(list)
        for p in common:
            parsed = parse_param(p)
            if parsed is None:
                continue
            k = key_fn(parsed)
            b_groups[k].append(before[p][0])
            a_groups[k].append(after[p][0])
        result = []
        for k in sorted(b_groups):
            mb = statistics.mean(b_groups[k])
            ma = statistics.mean(a_groups[k])
            result.append((k, mb, ma, len(b_groups[k])))
        return result

    print(f"# rel_l2 comparison report")
    print(f"# before: {len(before)} entries")
    print(f"# after:  {len(after)} entries")
    print(f"# common: {len(common)} entries")

    # --- by dtype ---
    rows = []
    for (dtype,), mb, ma, n in grouped(lambda p: (p[3],)):
        rows.append([dtype, str(n), f"{mb:.6f}", f"{ma:.6f}", f"{ma-mb:+.6f}", f"{pct(mb, ma):+.2f}%"])
    emit_compare_table(
        "Mean rel_l2 by dtype",
        rows,
        ["dtype", "N", "before", "after", "delta", "% change"],
        [6, 4, 10, 10, 11, 10],
        fmt,
    )

    # --- by (op, dtype) ---
    rows = []
    for (op, dtype), mb, ma, n in grouped(lambda p: (p[1], p[3])):
        rows.append([op, dtype, str(n), f"{mb:.6f}", f"{ma:.6f}", f"{ma-mb:+.6f}", f"{pct(mb, ma):+.2f}%"])
    emit_compare_table(
        "Mean rel_l2 by (op, dtype)",
        rows,
        ["op", "dtype", "N", "before", "after", "delta", "% change"],
        [22, 6, 4, 10, 10, 11, 10],
        fmt,
    )

    # --- by (dtype, fidelity) ---
    rows = []
    for (dtype, fid), mb, ma, n in grouped(lambda p: (p[3], p[4])):
        rows.append([dtype, fid, str(n), f"{mb:.6f}", f"{ma:.6f}", f"{ma-mb:+.6f}", f"{pct(mb, ma):+.2f}%"])
    emit_compare_table(
        "Mean rel_l2 by (dtype, fidelity)",
        rows,
        ["dtype", "fidelity", "N", "before", "after", "delta", "% change"],
        [6, 8, 4, 10, 10, 11, 10],
        fmt,
    )

    # --- by (layer, dtype) ---
    rows = []
    for (layer, dtype), mb, ma, n in grouped(lambda p: (p[0], p[3])):
        rows.append([layer, dtype, str(n), f"{mb:.6f}", f"{ma:.6f}", f"{ma-mb:+.6f}", f"{pct(mb, ma):+.2f}%"])
    emit_compare_table(
        "Mean rel_l2 by (layer, dtype)",
        rows,
        ["layer", "dtype", "N", "before", "after", "delta", "% change"],
        [10, 6, 4, 10, 10, 11, 10],
        fmt,
    )

    # --- top movers ---
    deltas = []
    for p in common:
        rb = before[p][0]
        ra = after[p][0]
        if rb > 0:
            deltas.append((pct(rb, ra), p, rb, ra))

    rows = [[f"{p:+.2f}%", f"{rb:.6f}", f"{ra:.6f}", k] for p, k, rb, ra in sorted(deltas)[:top]]
    emit_compare_table(
        f"Top {top} improvements (largest rel_l2 reduction)",
        rows,
        ["% change", "before", "after", "test"],
        [10, 10, 10, 70],
        fmt,
    )

    rows = [[f"{p:+.2f}%", f"{rb:.6f}", f"{ra:.6f}", k] for p, k, rb, ra in sorted(deltas, reverse=True)[:top]]
    emit_compare_table(
        f"Top {top} regressions (largest rel_l2 increase)",
        rows,
        ["% change", "before", "after", "test"],
        [10, 10, 10, 70],
        fmt,
    )

    improved = sum(1 for p, *_ in deltas if p < -0.1)
    regressed = sum(1 for p, *_ in deltas if p > 0.1)
    unchanged = sum(1 for p, *_ in deltas if -0.1 <= p <= 0.1)
    print()
    print(f"# Counts (>0.1% threshold): improved={improved}  regressed={regressed}  unchanged={unchanged}")


# ---------------------------------------------------------------------------
# Single-run mode
# ---------------------------------------------------------------------------


def report_single(data: Dict, top: int, fmt: str) -> None:
    print(f"# rel_l2 single-run report")
    print(f"# entries: {len(data)}")

    def grouped(key_fn) -> List[Tuple[Tuple, float, float, float, int]]:
        groups = defaultdict(list)
        for param, (rel_l2, _) in data.items():
            parsed = parse_param(param)
            if parsed is None:
                continue
            groups[key_fn(parsed)].append(rel_l2)
        result = []
        for k in sorted(groups):
            vals = groups[k]
            result.append((k, statistics.mean(vals), min(vals), max(vals), len(vals)))
        return result

    # by dtype
    rows = []
    for (dtype,), mean, mn, mx, n in grouped(lambda p: (p[3],)):
        rows.append([dtype, str(n), f"{mean:.6f}", f"{mn:.6f}", f"{mx:.6f}"])
    emit_single_table(
        "rel_l2 by dtype",
        rows,
        ["dtype", "N", "mean", "min", "max"],
        [6, 4, 10, 10, 10],
        fmt,
    )

    # by (op, dtype)
    rows = []
    for (op, dtype), mean, mn, mx, n in grouped(lambda p: (p[1], p[3])):
        rows.append([op, dtype, str(n), f"{mean:.6f}", f"{mn:.6f}", f"{mx:.6f}"])
    emit_single_table(
        "rel_l2 by (op, dtype)",
        rows,
        ["op", "dtype", "N", "mean", "min", "max"],
        [22, 6, 4, 10, 10, 10],
        fmt,
    )

    # worst cases
    worst = sorted(
        ((rel_l2, p) for p, (rel_l2, _) in data.items() if rel_l2 == rel_l2),  # filter NaN
        reverse=True,
    )[:top]
    rows = [[f"{rl:.6f}", k] for rl, k in worst]
    emit_single_table(
        f"Top {top} worst rel_l2",
        rows,
        ["rel_l2", "test"],
        [10, 80],
        fmt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path, help="JSONL from baseline run (or only run if --single)")
    parser.add_argument("after", type=Path, nargs="?", help="JSONL from comparison run (omit for single-run report)")
    parser.add_argument("--top", type=int, default=10, help="Number of top movers / worst cases to show (default: 10)")
    parser.add_argument("--format", choices=["text", "markdown"], default="text", help="Output format")
    args = parser.parse_args()

    before = load_jsonl(args.before)
    if args.after is None:
        report_single(before, top=args.top, fmt=args.format)
    else:
        after = load_jsonl(args.after)
        report_compare(before, after, top=args.top, fmt=args.format)


if __name__ == "__main__":
    main()
