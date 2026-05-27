#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Compare rel_l2 / pcc metrics between two JSONL files produced by
REL_L2_OUTPUT runs of test_matmul_gpt_oss_120b.py or test_layer_gpt_oss_120b.py.

Each JSONL line is: {"test_id": ..., "rel_l2": ..., "pcc": ...}

Usage:
    python scripts/compare_rel_l2.py before.jsonl after.jsonl

    # Only show cases where rel_l2 changed by more than a threshold:
    python scripts/compare_rel_l2.py before.jsonl after.jsonl --min-delta 0.01

    # Filter to a substring in test_id:
    python scripts/compare_rel_l2.py before.jsonl after.jsonl --filter "self_attn_q_proj"

    # Sort by delta (default), pcc, or rel_l2:
    python scripts/compare_rel_l2.py before.jsonl after.jsonl --sort-by delta
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> dict:
    """Load JSONL into dict keyed by test_id (last entry wins on dup)."""
    result = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: {path}:{line_no} skipping malformed line: {e}", file=sys.stderr)
                continue
            tid = entry.get("test_id")
            if tid:
                result[tid] = entry
    return result


def fmt_delta(d: float, width: int = 10) -> str:
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.6f}".rjust(width)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path, help="JSONL from baseline run")
    parser.add_argument("after", type=Path, help="JSONL from comparison run")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Only show rows where |after.rel_l2 - before.rel_l2| > this (default: 0)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Substring filter on test_id")
    parser.add_argument("--sort-by", choices=["delta", "rel_l2_after", "rel_l2_before", "pcc_after", "test_id"],
                        default="delta",
                        help="Sort key (default: delta)")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse sort order")
    args = parser.parse_args()

    before = load_jsonl(args.before)
    after = load_jsonl(args.after)

    before_keys = set(before.keys())
    after_keys = set(after.keys())
    common = before_keys & after_keys
    only_before = before_keys - after_keys
    only_after = after_keys - before_keys

    rows = []
    for tid in common:
        if args.filter and args.filter not in tid:
            continue
        b = before[tid]
        a = after[tid]
        delta_rel_l2 = a["rel_l2"] - b["rel_l2"]
        delta_pcc = a["pcc"] - b["pcc"]
        if abs(delta_rel_l2) <= args.min_delta:
            continue
        rows.append({
            "test_id": tid,
            "rel_l2_before": b["rel_l2"],
            "rel_l2_after": a["rel_l2"],
            "delta": delta_rel_l2,
            "pcc_before": b["pcc"],
            "pcc_after": a["pcc"],
            "delta_pcc": delta_pcc,
        })

    sort_key = {
        "delta": lambda r: abs(r["delta"]),
        "rel_l2_after": lambda r: r["rel_l2_after"],
        "rel_l2_before": lambda r: r["rel_l2_before"],
        "pcc_after": lambda r: r["pcc_after"],
        "test_id": lambda r: r["test_id"],
    }[args.sort_by]
    rows.sort(key=sort_key, reverse=not args.reverse if args.sort_by != "test_id" else args.reverse)

    print(f"# Compare: {args.before.name} vs {args.after.name}")
    print(f"# common={len(common)}  only_before={len(only_before)}  only_after={len(only_after)}")
    if args.min_delta > 0:
        print(f"# filtered to |Δrel_l2| > {args.min_delta}")
    if args.filter:
        print(f"# filtered to test_id contains '{args.filter}'")
    print()

    if not rows:
        print("(no rows match filters)")
    else:
        print(f"{'rel_l2_before':>14}  {'rel_l2_after':>14}  {'Δrel_l2':>11}  "
              f"{'pcc_before':>10}  {'pcc_after':>10}  {'Δpcc':>10}  test_id")
        print("-" * 120)
        for r in rows:
            print(f"{r['rel_l2_before']:>14.6f}  {r['rel_l2_after']:>14.6f}  {fmt_delta(r['delta'], 11)}  "
                  f"{r['pcc_before']:>10.6f}  {r['pcc_after']:>10.6f}  {fmt_delta(r['delta_pcc'], 10)}  "
                  f"{r['test_id']}")

    # Summary
    if rows:
        improved = sum(1 for r in rows if r["delta"] < 0)
        regressed = sum(1 for r in rows if r["delta"] > 0)
        unchanged = sum(1 for r in rows if r["delta"] == 0)
        max_imp = min(rows, key=lambda r: r["delta"])
        max_reg = max(rows, key=lambda r: r["delta"])
        print()
        print(f"# Summary: improved={improved}  regressed={regressed}  unchanged={unchanged}")
        print(f"# Max improvement: {max_imp['delta']:+.6f}  ({max_imp['test_id']})")
        print(f"# Max regression:  {max_reg['delta']:+.6f}  ({max_reg['test_id']})")

    if only_before or only_after:
        print()
        print(f"# WARNING: test_id sets differ")
        if only_before:
            print(f"#   {len(only_before)} only in {args.before.name} (showing first 3):")
            for tid in list(only_before)[:3]:
                print(f"#     {tid}")
        if only_after:
            print(f"#   {len(only_after)} only in {args.after.name} (showing first 3):")
            for tid in list(only_after)[:3]:
                print(f"#     {tid}")


if __name__ == "__main__":
    main()
