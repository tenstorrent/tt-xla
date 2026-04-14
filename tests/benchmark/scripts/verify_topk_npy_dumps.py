#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Verify topk .npy dumps produced by TT_RUNTIME_OP_TENSOR_TRACE_TOPK_DUMP_DIR.

For each topk invocation found in the dump directory, loads the input, values,
and indices tensors per device, runs torch.topk as golden reference, and checks:

  1. Values PCC (Pearson correlation) against golden
  2. No duplicate indices per row
  3. Gathered-values cosine similarity (indices select the right input elements)
  4. All-zero detection (flags fully-zero outputs)

Usage:
    python scripts/verify_topk_npy_dumps.py /tmp/topk_dump
    python scripts/verify_topk_npy_dumps.py /tmp/topk_dump --pcc-threshold 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_BENCH_ROOT))

from benchmark.utils import compute_pcc  # noqa: E402


def _discover_topk_ops(dump_dir: Path) -> list[int]:
    """Find all topk op sequence numbers from meta files."""
    seqs = set()
    for meta in dump_dir.glob("topk_*_meta.json"):
        stem = meta.stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                seqs.add(int(parts[1]))
            except ValueError:
                pass
    return sorted(seqs)


def _discover_devices(dump_dir: Path, seq: int, tensor_name: str) -> list[int]:
    """Find all device indices for a given topk op and tensor name."""
    devs = set()
    for f in dump_dir.glob(f"topk_{seq}_{tensor_name}_dev*.npy"):
        stem = f.stem
        dev_str = stem.split("_dev")[-1]
        try:
            devs.add(int(dev_str))
        except ValueError:
            pass
    return sorted(devs)


def _load_npy(dump_dir: Path, seq: int, name: str, dev: int) -> np.ndarray | None:
    p = dump_dir / f"topk_{seq}_{name}_dev{dev}.npy"
    if not p.exists():
        return None
    return np.load(p)


def verify_topk_op(
    dump_dir: Path,
    seq: int,
    meta: dict,
    pcc_threshold: float = 0.99,
    cos_threshold: float = 0.99,
) -> tuple[int, int]:
    """Verify one topk invocation. Returns (pass_count, fail_count)."""
    k = meta["k"]
    dim = meta.get("dim", -1)
    largest = meta.get("largest", True)

    input_devs = _discover_devices(dump_dir, seq, "input")
    values_devs = _discover_devices(dump_dir, seq, "values")
    indices_devs = _discover_devices(dump_dir, seq, "indices")

    all_devs = sorted(set(input_devs) | set(values_devs) | set(indices_devs))
    if not all_devs:
        print(f"  [SKIP] no device tensors found")
        return 0, 0

    passes = 0
    fails = 0

    for dev in all_devs:
        prefix = f"  dev={dev}"
        inp_np = _load_npy(dump_dir, seq, "input", dev)
        val_np = _load_npy(dump_dir, seq, "values", dev)
        idx_np = _load_npy(dump_dir, seq, "indices", dev)

        if inp_np is None:
            print(f"{prefix} | SKIP | no input tensor")
            continue

        inp = torch.from_numpy(inp_np).float()

        # Golden reference
        golden_values, golden_indices = torch.topk(inp, k, dim=dim, largest=largest)

        # --- Check values ---
        if val_np is not None:
            dev_values = torch.from_numpy(val_np).float()
            all_zero = (dev_values == 0).all().item()

            if all_zero:
                print(f"{prefix} | FAIL | values ALL ZERO | shape={list(dev_values.shape)}")
                fails += 1
                continue

            try:
                pcc = compute_pcc(golden_values, dev_values)
            except Exception as e:
                pcc = float("nan")
                print(f"{prefix} | FAIL | values PCC error: {e}")
                fails += 1
                continue

            pcc_ok = pcc > pcc_threshold
        else:
            pcc = float("nan")
            pcc_ok = True  # no values to check

        # --- Check indices ---
        cos_sim = float("nan")
        dup_ok = True
        cos_ok = True

        if idx_np is not None:
            dev_indices = torch.from_numpy(idx_np).long()
            all_zero_idx = (dev_indices == 0).all().item()

            if all_zero_idx and k > 1:
                print(f"{prefix} | FAIL | indices ALL ZERO | shape={list(dev_indices.shape)}")
                fails += 1
                continue

            # Duplicate check
            for r in range(dev_indices.shape[0]):
                row = dev_indices[r]
                if row.unique().numel() != row.numel():
                    dup_ok = False
                    break

            # Gathered cosine similarity
            try:
                gathered = torch.gather(inp, dim, dev_indices)
                cos_sim = torch.nn.functional.cosine_similarity(
                    gathered.flatten().unsqueeze(0).float(),
                    golden_values.flatten().unsqueeze(0).float(),
                ).item()
                cos_ok = cos_sim > cos_threshold
            except Exception as e:
                cos_ok = False
                print(f"{prefix} | WARN | gather/cosine error: {e}")

        ok = pcc_ok and dup_ok and cos_ok
        status = "PASS" if ok else "FAIL"
        parts = [f"{prefix} | {status}"]
        if val_np is not None:
            parts.append(f"pcc={pcc:.6f}")
        if idx_np is not None:
            parts.append(f"cos={cos_sim:.6f}")
            if not dup_ok:
                parts.append("DUP_INDICES")
        if val_np is not None:
            parts.append(f"shape={list(dev_values.shape)}")

        print(" | ".join(parts))

        if ok:
            passes += 1
        else:
            fails += 1

    return passes, fails


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path, help="Directory with topk_*.npy files")
    parser.add_argument("--pcc-threshold", type=float, default=0.99)
    parser.add_argument("--cos-threshold", type=float, default=0.99)
    args = parser.parse_args()

    if not args.dump_dir.is_dir():
        print(f"Error: {args.dump_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    seqs = _discover_topk_ops(args.dump_dir)
    if not seqs:
        print(f"No topk dumps found in {args.dump_dir}")
        sys.exit(1)

    print(f"Found {len(seqs)} topk invocation(s) in {args.dump_dir}\n")

    total_pass = 0
    total_fail = 0

    for seq in seqs:
        meta_path = args.dump_dir / f"topk_{seq}_meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
        else:
            meta = {"k": 4, "dim": -1, "largest": True, "sorted": True}

        input_devs = _discover_devices(args.dump_dir, seq, "input")
        values_devs = _discover_devices(args.dump_dir, seq, "values")
        indices_devs = _discover_devices(args.dump_dir, seq, "indices")

        print(
            f"topk #{seq} | k={meta.get('k')} dim={meta.get('dim', -1)} "
            f"| devices: input={len(input_devs)} values={len(values_devs)} "
            f"indices={len(indices_devs)}"
        )

        p, f = verify_topk_op(
            args.dump_dir, seq, meta,
            pcc_threshold=args.pcc_threshold,
            cos_threshold=args.cos_threshold,
        )
        total_pass += p
        total_fail += f
        print()

    print(f"{'='*60}")
    print(f"TOTAL: {total_pass} passed, {total_fail} failed")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
