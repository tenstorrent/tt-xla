#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Convert all new-format (ttnn_*) .npy dump files to readable CSVs.

Usage:
    python dump_to_csv.py [DUMP_DIR] [--devices 0,1,2] [--all-devices] [--pipeline-only]

Defaults to device 0 only.  Output goes to DUMP_DIR/readable/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def discover_metas(dump_dir: Path) -> list[dict]:
    metas = []
    for f in sorted(dump_dir.glob("ttnn_*_meta.json")):
        with open(f) as fp:
            m = json.load(fp)
        if "params" not in m:
            continue
        m["_meta_path"] = str(f)
        m["_base"] = str(f).rsplit("_meta.json", 1)[0]
        metas.append(m)
    return sorted(metas, key=lambda x: x["op_seq"])


def filter_pipeline(ops: list[dict]) -> list[dict]:
    topk_seqs = [m["op_seq"] for m in ops if m.get("mlir_op") == "ttnn.topk"]
    if not topk_seqs:
        return ops
    scatter_seqs = sorted(
        m["op_seq"] for m in ops if m.get("mlir_op") == "ttnn.scatter"
    )
    ranges = []
    for tk in topk_seqs:
        next_topk = min((t for t in topk_seqs if t > tk), default=float("inf"))
        pipeline_scatters = [s for s in scatter_seqs if tk < s < next_topk]
        last_scatter = max(pipeline_scatters) if pipeline_scatters else tk
        ranges.append((tk, last_scatter))

    def _in_pipeline(seq):
        return any(lo <= seq <= hi for lo, hi in ranges)

    return [m for m in ops if _in_pipeline(m["op_seq"])]


def find_npy_files(base: str, devices: set[int] | None) -> list[tuple[str, str, int]]:
    """Return [(tag, path, dev_id), ...] for npy files matching the base prefix."""
    parent = Path(base).parent
    stem = Path(base).name
    results = []
    for f in sorted(parent.iterdir()):
        name = f.name
        if not name.startswith(stem + "_") or not name.endswith(".npy"):
            continue
        rest = name[len(stem) + 1 : -len(".npy")]
        if "_dev" not in rest:
            continue
        tag, dev_str = rest.rsplit("_dev", 1)
        try:
            dev_id = int(dev_str)
        except ValueError:
            continue
        if devices is not None and dev_id not in devices:
            continue
        results.append((tag, str(f), dev_id))
    return results


def npy_to_csv(npy_path: str, csv_path: str):
    arr = np.load(npy_path)
    if arr.ndim == 0:
        with open(csv_path, "w") as fp:
            fp.write(f"{arr.item()}\n")
        return
    if arr.ndim == 1:
        np.savetxt(csv_path, arr, delimiter=",", fmt=_fmt(arr.dtype))
    else:
        flat = arr.reshape(-1) if arr.ndim > 2 else None
        if arr.ndim == 2:
            np.savetxt(csv_path, arr, delimiter=",", fmt=_fmt(arr.dtype))
        else:
            with open(csv_path, "w") as fp:
                fp.write(f"# shape: {list(arr.shape)}\n")
                rows = arr.reshape(-1, arr.shape[-1])
                for row in rows:
                    fp.write(",".join(_fmtval(v, arr.dtype) for v in row) + "\n")


def _fmt(dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return "%d"
    return "%.6g"


def _fmtval(v, dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return str(int(v))
    return f"{v:.6g}"


def main():
    parser = argparse.ArgumentParser(description="Convert .npy dumps to CSV")
    parser.add_argument("dump_dir", nargs="?",
                        default="tests/benchmark/modules/gpt_oss_input_sharding_dbg/topk_dump",
                        help="Path to dump directory")
    parser.add_argument("--devices", type=str, default="0",
                        help="Comma-separated device IDs (default: 0)")
    parser.add_argument("--all-devices", action="store_true",
                        help="Convert all devices")
    parser.add_argument("--pipeline-only", action="store_true",
                        help="Only convert pipeline ops (topk→scatter)")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    if not dump_dir.is_dir():
        print(f"ERROR: {dump_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    devices = None if args.all_devices else set(int(d) for d in args.devices.split(","))

    out_dir = dump_dir / "readable"
    out_dir.mkdir(exist_ok=True)

    ops = discover_metas(dump_dir)
    if args.pipeline_only:
        ops = filter_pipeline(ops)

    print(f"Found {len(ops)} ops, devices={devices or 'all'}")
    print(f"Output: {out_dir}/\n")

    summary_lines = []
    total_files = 0

    for m in ops:
        base = m["_base"]
        seq = m["op_seq"]
        mlir_op = m.get("mlir_op", "?")
        loc = m.get("loc", "")
        params = m.get("params", {})

        npy_files = find_npy_files(base, devices)
        if not npy_files:
            continue

        label = f"seq{seq:>4d}  {mlir_op:<25s}  loc={loc}"
        summary_lines.append(label)
        print(f"  {label}  ({len(npy_files)} files)")

        for tag, npy_path, dev_id in npy_files:
            stem = Path(base).name
            csv_name = f"{stem}_{tag}_dev{dev_id}.csv"
            csv_path = out_dir / csv_name
            try:
                npy_to_csv(npy_path, str(csv_path))
                total_files += 1
            except Exception as e:
                print(f"    WARN: {csv_name}: {e}")

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as fp:
        fp.write(f"# Readable dump summary — {len(ops)} ops, "
                 f"devices={devices or 'all'}\n\n")
        for line in summary_lines:
            fp.write(line + "\n")
        fp.write(f"\nParams per op:\n")
        for m in ops:
            seq = m["op_seq"]
            mlir_op = m.get("mlir_op", "?")
            params = m.get("params", {})
            fp.write(f"  seq{seq}: {mlir_op} {json.dumps(params)}\n")

    print(f"\nDone: {total_files} CSV files written to {out_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
