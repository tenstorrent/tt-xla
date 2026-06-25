#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Parse a pytest ``--runxfail --tb=line`` log of test_matmul_mp.py into a
markdown PCC report.

Usage:
    python build_report.py runxfail.log > pcc_report.md

The log must come from a run with ``--runxfail`` (all xfailed tests become
real failures, exposing their PCC values) and ``--tb=line`` (each failure
prints exactly one AssertionError line containing ``Calculated: pcc=...``).
Failures and PCC lines are matched positionally by execution order, so the
script breaks if either pytest reorders that output.
"""

import re
import sys
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "runxfail.log"

ID_RE = re.compile(r"FAILED .*test_matmul_mp\[([^]]+)\]")
PCC_RE = re.compile(r"Calculated: pcc=([0-9.]+?)\.\s")
TID_RE = re.compile(
    r"\((\d+),(\d+),(\d+)\)x\((\d+),(\d+)\)-"
    r"(FROM_\w+)-mp_opt(\d+)_(\w+?)_fp32acc(true|false)_(\w+)$"
)


def parse_log(path):
    with open(path) as f:
        text = f.read()
    ids = ID_RE.findall(text)
    pccs = [float(p) for p in PCC_RE.findall(text)]
    if len(ids) != len(pccs):
        sys.exit(
            f"id/pcc length mismatch: {len(ids)} ids vs {len(pccs)} pccs"
        )
    rows = []
    for tid, pcc in zip(ids, pccs):
        m = TID_RE.match(tid)
        if not m:
            sys.exit(f"unparseable id: {tid}")
        b, s, k, k2, n, src, opt, wd, fp32, mf = m.groups()
        rows.append(
            {
                "shape": f"({b},{s},{k})x({k2},{n})",
                "src": src.replace("FROM_", ""),
                "opt": int(opt),
                "wd": wd,
                "fp32": fp32 == "true",
                "mf": mf,
                "pcc": pcc,
            }
        )
    return rows


def emit(rows):
    out = []
    e = out.append
    e("# matmul_mp PCC report\n")
    e(f"Failures: **{len(rows)}**. Distinct PCC values: **{len(set(r['pcc'] for r in rows))}**.\n")

    shapes = sorted({r["shape"] for r in rows})

    e("## Summary by shape\n")
    e("| Shape | Fails | PCC min | PCC max |")
    e("|---|---|---|---|")
    for shape in shapes:
        sr = [r for r in rows if r["shape"] == shape]
        pccs = [r["pcc"] for r in sr]
        e(f"| `{shape}` | {len(sr)} | {min(pccs):.4f} | {max(pccs):.4f} |")
    e("")

    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["shape"], r["opt"], r["fp32"], r["wd"], r["mf"])].append(
            (r["src"], r["pcc"])
        )

    for shape in shapes:
        sr = [r for r in rows if r["shape"] == shape]
        e(f"## `{shape}` ({len(sr)} fails)\n")
        e("| opt | fp32acc | wd | mf | ANOTHER_OP | HOST |")
        e("|---|---|---|---|---|---|")
        keys = sorted({(r["opt"], r["fp32"], r["wd"], r["mf"]) for r in sr})
        for opt, fp32, wd, mf in keys:
            entries = buckets[(shape, opt, fp32, wd, mf)]
            another = next((p for s, p in entries if s == "ANOTHER_OP"), None)
            host = next((p for s, p in entries if s == "HOST"), None)
            a = f"{another:.6f}" if another is not None else "—"
            h = f"{host:.6f}" if host is not None else "—"
            e(f"| {opt} | {fp32} | {wd} | {mf} | {a} | {h} |")
        e("")

    return "\n".join(out)


if __name__ == "__main__":
    print(emit(parse_log(LOG)))
