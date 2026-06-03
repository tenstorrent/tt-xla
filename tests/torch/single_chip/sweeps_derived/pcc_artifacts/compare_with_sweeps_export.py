#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-check the test_matmul_mp.py snapshots against a sweeps CSV export.

The CSV (e.g. `sweeps_export.csv`) is a pivot of sweeps `MIN(pcc)` per
(input_source, shape, run_id) row, with 20 columns of
`MIN(pcc) <opt> <fp32acc> <math_fidelity> <weight_dtype>`. It comes from
the sweeps dashboard and snapshots the entire matmul_mp suite across
multiple CI runs.

This script does three things:
1. Verifies that the shapes in `test_matmul_mp.py::_SHAPE_PAIRS` are
   present in the CSV and reports their per-run PCC profile.
2. Flags new regressions: (src, shape) pairs catastrophic only in the
   latest run.
3. Flags persistent bugs: catastrophic in every run in the CSV.

Run:
    python3 compare_with_sweeps_export.py [path/to/sweeps_export.csv]
"""

import csv
import re
import sys
from collections import defaultdict

CSV_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else f"{__file__.rsplit('/', 1)[0]}/sweeps_export.csv"
)

# Catastrophic threshold — PCC below this counts as a real failure (not noise).
CATASTROPHIC_PCC = 0.7

# The shapes covered by test_matmul_mp.py. Keep in sync with the test file
# (currently `_SHAPE_PAIRS`).
MY_SHAPES = [
    "((32, 128, 1024), (1024, 2048))",
    "((32, 128, 2304), (2304, 1024))",
    "((32, 128, 2560), (2560, 1024))",
    "((32, 128, 4864), (4864, 896))",
]


def parse_label(label):
    """Row label: '<input_source> <shape> <run_id>'."""
    m = re.match(r"(FROM_\w+)\s+(\(\(.*\)\))\s+(\d+)", label)
    return m.groups() if m else (None, None, None)


def parse_cols(header):
    """Header columns: 'MIN(pcc) <opt> <True|False> <mf> <wd>'."""
    cols = []
    for h in header[1:]:
        parts = h.split()
        cols.append((int(parts[1]), parts[2] == "True", parts[3], parts[4]))
    return cols


def main():
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    cols = parse_cols(header)

    # Index: (src, shape) -> {run -> [(opt, fp32, mf, wd, pcc), ...]}
    grid = defaultdict(lambda: defaultdict(list))
    runs = set()
    for row in rows:
        src, shape, run = parse_label(row[0])
        if not src:
            continue
        runs.add(run)
        for i, val in enumerate(row[1:]):
            if not val:
                continue
            grid[(src, shape)][run].append((*cols[i], float(val)))

    runs_sorted = sorted(runs)
    latest = runs_sorted[-1]
    all_runs = set(runs_sorted)
    print(f"CSV: {len(rows)} rows, {len(runs)} runs: {runs_sorted}")
    print(f"Catastrophic threshold: PCC < {CATASTROPHIC_PCC}")
    print(f"Latest run: {latest}\n")

    # (1) Per-shape profile for the test grid
    print("=" * 78)
    print("Test grid shapes — per-run min PCC at opt=2")
    print("=" * 78)
    for shape in MY_SHAPES:
        print(f"\n{shape}")
        for src in ("FROM_ANOTHER_OP", "FROM_HOST"):
            line = f"  {src:<18}"
            for run in runs_sorted:
                cells = grid[(src, shape)][run]
                opt2 = [pcc for opt, _, _, _, pcc in cells if opt == 2]
                if not opt2:
                    line += f"  {run}: (no data)"
                else:
                    mark = "✗" if min(opt2) < CATASTROPHIC_PCC else "✓"
                    line += f"  {run}={min(opt2):.4f} {mark}"
            print(line)

    # (2) Catastrophic classification across the whole CSV
    catastrophic = defaultdict(set)  # (src, shape) -> set of runs where catastrophic
    for (src, shape), per_run in grid.items():
        for run, cells in per_run.items():
            for opt, _, _, _, pcc in cells:
                if opt == 2 and pcc < CATASTROPHIC_PCC:
                    catastrophic[(src, shape)].add(run)
                    break

    persistent = {k for k, v in catastrophic.items() if v == all_runs}
    only_latest = {k for k, v in catastrophic.items() if v == {latest}}

    print("\n" + "=" * 78)
    print(f"New regressions (catastrophic only in {latest})")
    print("=" * 78)
    for src, shape in sorted(only_latest):
        in_grid = "  ← in test grid" if shape in MY_SHAPES else ""
        print(f"  {src:<18} {shape}{in_grid}")

    print("\n" + "=" * 78)
    print(f"Persistent bugs (catastrophic in all {len(all_runs)} runs)")
    print("=" * 78)
    for src, shape in sorted(persistent):
        in_grid = "  ← in test grid" if shape in MY_SHAPES else ""
        print(f"  {src:<18} {shape}{in_grid}")

    # (3) Cross-validate latest-run PCC against the realistic snapshot
    print("\n" + "=" * 78)
    print("PCC cross-check vs realistic snapshot (sample, latest run only)")
    print("=" * 78)
    try:
        with open(
            f"{__file__.rsplit('/', 1)[0]}/runxfail_realistic.log"
        ) as f:
            log = f.read()
        ids = re.findall(r"FAILED .*test_matmul_mp\[([^\]]+)\]", log)
        pccs = [float(p) for p in re.findall(r"Calculated: pcc=([0-9.]+?)\.\s", log)]
        ttxla = dict(zip(ids, pccs))
    except FileNotFoundError:
        print("  runxfail_realistic.log not found — skipping cross-check")
        return

    print(f"  {'shape':<40} {'tt-xla':>10} {'sweeps':>10} {'Δ':>8}")
    seen = set()
    for tid, ttxla_pcc in sorted(ttxla.items()):
        # tid: "<shape>-FROM_ANOTHER_OP-mp_opt2_..."
        parts = tid.split("-", 1)
        shape_compact = parts[0]  # e.g. (32,128,1024)x(1024,2048)
        # Map "(32,128,1024)x(1024,2048)" -> "((32, 128, 1024), (1024, 2048))"
        m = re.match(r"\((\d+),(\d+),(\d+)\)x\((\d+),(\d+)\)", shape_compact)
        if not m:
            continue
        b, s, k, k2, n = m.groups()
        shape_full = f"(({b}, {s}, {k}), ({k2}, {n}))"
        if shape_full in seen:
            continue
        seen.add(shape_full)
        # Sweeps min opt=2 PCC for this shape, FROM_ANOTHER_OP, latest run
        cells = grid[("FROM_ANOTHER_OP", shape_full)][latest]
        opt2 = [pcc for opt, _, _, _, pcc in cells if opt == 2]
        if not opt2:
            continue
        sweeps_pcc = min(opt2)
        print(f"  {shape_full:<40} {ttxla_pcc:>10.4f} {sweeps_pcc:>10.4f} {sweeps_pcc - ttxla_pcc:>+8.4f}")


if __name__ == "__main__":
    main()
