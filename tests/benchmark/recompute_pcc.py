# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Recompute per-output PCC with float64 accumulation + chunking.

The initial run produced PCC > 1 on huge (13B-element) outputs because it
summed in fp32. Pearson correlation is bounded in [-1, 1]; exceeding that
range is proof of accumulation error. This script works off the saved
device_outputs.pt / bypass_outputs.pt so no device rerun is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def _chunked_sum_f64(x: torch.Tensor, chunk: int = 1_000_000) -> float:
    """Sum x in float64 in chunks (x may be bf16/fp32 and huge)."""
    total = 0.0
    flat = x.reshape(-1)
    n = flat.numel()
    for start in range(0, n, chunk):
        total += flat[start : start + chunk].to(torch.float64).sum().item()
    return total


def _dedupe_replicated_leading_dim(t: torch.Tensor) -> torch.Tensor:
    """If dim 0 is fully replicated (all slices identical), collapse to one slice.

    The gather step stacks 32 per-chip shards along dim 0. When the original
    output was replicated across some axis, multiple shards are identical —
    including them all inflates the element count without adding information.
    """
    if t.ndim == 0 or t.shape[0] == 1:
        return t
    first = t[0]
    for i in range(1, t.shape[0]):
        # Use a small tolerance for bf16 noise.
        if not torch.allclose(first, t[i], atol=0.0, rtol=0.0):
            return t
    return first


def compute_pcc_f64(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    n = a.numel()
    if n == 0:
        return float("nan")
    # Means in fp64 (chunked).
    mean_a = _chunked_sum_f64(a) / n
    mean_b = _chunked_sum_f64(b) / n
    # sum((a - mean_a) * (b - mean_b)), sum((a - mean_a)^2), sum((b - mean_b)^2)
    # all in fp64 without materialising the centred tensors.
    num = 0.0
    va = 0.0
    vb = 0.0
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    chunk = 1_000_000
    for start in range(0, n, chunk):
        ac = flat_a[start : start + chunk].to(torch.float64) - mean_a
        bc = flat_b[start : start + chunk].to(torch.float64) - mean_b
        num += (ac * bc).sum().item()
        va += (ac * ac).sum().item()
        vb += (bc * bc).sum().item()
    denom = (va ** 0.5) * (vb ** 0.5)
    if denom == 0.0:
        return float("nan")
    return num / denom


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <generated_dir>", file=sys.stderr)
        sys.exit(2)
    gen = Path(sys.argv[1]).resolve()
    dev_path = gen / "device_outputs.pt"
    byp_path = gen / "bypass_outputs.pt"
    print(f"Loading device outputs: {dev_path}")
    dev_tensors = torch.load(dev_path, weights_only=True, map_location="cpu")
    print(f"Loading bypass outputs: {byp_path}")
    byp_tensors = torch.load(byp_path, weights_only=True, map_location="cpu")

    if len(dev_tensors) != len(byp_tensors):
        sys.exit(f"Output count mismatch: dev={len(dev_tensors)} byp={len(byp_tensors)}")

    print("\n=== Per-output PCC (device vs CPU bypass), fp64 ===")
    pccs = []
    for i, (d, b) in enumerate(zip(dev_tensors, byp_tensors)):
        d2 = _dedupe_replicated_leading_dim(d)
        b2 = _dedupe_replicated_leading_dim(b)
        if d2.shape != b2.shape:
            print(f"  [{i}] shape mismatch post-dedup: dev={tuple(d2.shape)} byp={tuple(b2.shape)} -- SKIPPING")
            continue
        pcc = compute_pcc_f64(d2, b2)
        pccs.append(pcc)
        print(f"  [{i}] shape={tuple(d2.shape)} ({d2.numel():,} elems) PCC={pcc:+.6f}")

    if pccs:
        mean = sum(pccs) / len(pccs)
        worst = min(pccs)
        print(f"\nOverall mean PCC: {mean:+.6f}")
        print(f"Worst PCC:        {worst:+.6f}")


if __name__ == "__main__":
    main()
