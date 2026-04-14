#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare two TTNN perf-metrics JSON files (e.g. decode-only vs all-step sharding).

These files are emitted when ``ttnn_perf_metrics_enabled`` is on (see
``debug_gpt_oss_input_sharding.py --ttnn-perf-metrics``). They list **per-op
metadata** (operation name, MLIR ``location``, ``layout_info``, sharding flags)
— not numeric tensor dumps. Full input/output values for every device op would
require compiler/runtime instrumentation beyond this JSON.

From ``tests/benchmark``::

    python scripts/compare_ttnn_perf_sharding.py \\
        --left modules/gpt_oss_input_sharding_dbg/ttnn_perf/dec_perf_metrics_1.json \\
        --right modules/gpt_oss_input_sharding_dbg/ttnn_perf/all_perf_metrics_1.json

Lines worth attention are prefixed with ``>>>``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any]:
    with p.open() as f:
        return json.load(f)


def _norm_entry(e: dict[str, Any]) -> tuple[str, str, str, bool, bool]:
    return (
        str(e.get("operation", "")),
        str(e.get("location", "")),
        str(e.get("layout_info", "")),
        bool(e.get("is_sharded", False)),
        bool(e.get("has_system_memory", False)),
    )


def _collective_like(op: str) -> bool:
    needles = (
        "all_gather",
        "all_reduce",
        "reduce_scatter",
        "all_to_all",
        "mesh_shard",
        "mesh_partition",
        "scatter",
        "point_to_point",
        "p2p",
    )
    return any(n in op for n in needles)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", type=Path, required=True, help="e.g. dec graph JSON")
    ap.add_argument("--right", type=Path, required=True, help="e.g. all graph JSON")
    args = ap.parse_args()

    left = _load(args.left)
    right = _load(args.right)

    print(f"left : {args.left}")
    print(f"right: {args.right}")
    print()

    lb = left.get("operation_type_breakdown") or {}
    rb = right.get("operation_type_breakdown") or {}
    keys = sorted(set(lb) | set(rb))
    for k in keys:
        a, b = lb.get(k, 0), rb.get(k, 0)
        if a != b:
            mark = ">>> " if _collective_like(k) or abs(a - b) > max(a, b, 1) * 0.2 else ""
            print(f"{mark}operation_type_breakdown[{k!r}]: left={a} right={b} (delta {b - a:+d})")

    print()
    ls = left.get("summary") or {}
    rs = right.get("summary") or {}
    sk = sorted(set(ls) | set(rs))
    for k in sk:
        a, b = ls.get(k), rs.get(k)
        if a != b:
            print(f">>> summary[{k!r}]: left={a!r} right={b!r}")

    lsh = left.get("shardable_operations") or []
    rsh = right.get("shardable_operations") or []
    cl = Counter(_norm_entry(e) for e in lsh if isinstance(e, dict))
    cr = Counter(_norm_entry(e) for e in rsh if isinstance(e, dict))

    only_left = cl - cr
    only_right = cr - cl
    if only_left:
        print()
        print(f">>> {len(only_left)} shardable entry kinds only on left (sample up to 12):")
        for t, c in only_left.most_common(12):
            print(f"    x{c}  op={t[0]!r} loc={t[1]!r} sharded={t[3]}")
    if only_right:
        print()
        print(f">>> {len(only_right)} shardable entry kinds only on right (sample up to 12):")
        for t, c in only_right.most_common(12):
            print(f"    x{c}  op={t[0]!r} loc={t[1]!r} sharded={t[3]}")

    # Same (op, location) multiset but different layout or is_sharded between sides
    by_loc_op_l: dict[tuple[str, str], Counter[tuple[str, bool]]] = {}
    by_loc_op_r: dict[tuple[str, str], Counter[tuple[str, bool]]] = {}
    for e in lsh:
        if not isinstance(e, dict):
            continue
        key = (str(e.get("operation", "")), str(e.get("location", "")))
        by_loc_op_l.setdefault(key, Counter())
        by_loc_op_l[key][(str(e.get("layout_info", "")), bool(e.get("is_sharded", False)))] += 1
    for e in rsh:
        if not isinstance(e, dict):
            continue
        key = (str(e.get("operation", "")), str(e.get("location", "")))
        by_loc_op_r.setdefault(key, Counter())
        by_loc_op_r[key][(str(e.get("layout_info", "")), bool(e.get("is_sharded", False)))] += 1

    mismatches = 0
    for key in sorted(set(by_loc_op_l) | set(by_loc_op_r)):
        if by_loc_op_l.get(key) != by_loc_op_r.get(key):
            mismatches += 1
            if mismatches <= 20:
                op, loc = key
                mark = ">>> " if _collective_like(op) else ""
                print()
                print(f"{mark}shardable_operations differ for op={op!r} loc={loc!r}")
                print(f"    left:  {dict(by_loc_op_l.get(key, {}))}")
                print(f"    right: {dict(by_loc_op_r.get(key, {}))}")
    if mismatches > 20:
        print()
        print(f">>> ... and {mismatches - 20} more (op,location) groups with layout/shard differences")


if __name__ == "__main__":
    main()
