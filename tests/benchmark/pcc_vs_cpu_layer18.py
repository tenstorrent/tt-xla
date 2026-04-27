# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC device/bypass vs CPU reference (non-compiled bf16 torch model).

Inputs (under tests/benchmark/bfp4_layer18_output/):
  device_outputs.pt         list of (32, *shard_shape) tensors (stacked shards)
  bypass_outputs.pt         list of (32, *shard_shape) tensors (stacked shards)
  cpu_reference_outputs.pt  {"tensors": [...], "names": [...]} from run_cpu_reference_layer18.py

For each logical CPU reference tensor, locate the matching device/bypass
tensor by shape, reconstruct it from the 4x8 mesh of per-chip shards
(shard-dim detected by shape-diff of neighbor shards), then compute PCC
in float64 with chunked accumulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Mesh-aware reconstruction of a full logical tensor from a (32, *) stack.
# ---------------------------------------------------------------------------

MESH_ROWS = 4
MESH_COLS = 8


def _reconstruct_from_mesh(stacked: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Reconstruct the full logical tensor from a (32, *shard) mesh stack.

    stacked[i] for i in 0..31 is the shard held by chip i in a row-major
    (4 rows x 8 cols) mesh. The shard dims along each mesh axis are deduced
    from `target_shape` (the CPU-reference logical shape): dim d is sharded
    across rows if per_chip_shape[d] * MESH_ROWS == target_shape[d], across
    cols if * MESH_COLS == target_shape[d], replicated otherwise.
    """
    if stacked.shape[0] != MESH_ROWS * MESH_COLS:
        return stacked
    per_chip_shape = tuple(stacked.shape[1:])
    if len(per_chip_shape) != len(target_shape):
        return stacked

    row_dim = None
    col_dim = None
    for d in range(len(per_chip_shape)):
        pd, td = per_chip_shape[d], target_shape[d]
        if pd == td:
            continue    # dim already full — replicated along both axes
        if pd * MESH_ROWS == td and row_dim is None:
            row_dim = d
        elif pd * MESH_COLS == td and col_dim is None:
            col_dim = d

    grid = [[stacked[r * MESH_COLS + c] for c in range(MESH_COLS)] for r in range(MESH_ROWS)]
    row_tensors = []
    for r in range(MESH_ROWS):
        if col_dim is not None:
            row_tensors.append(torch.cat(grid[r], dim=col_dim))
        else:
            row_tensors.append(grid[r][0])    # col-replicated
    if row_dim is not None:
        return torch.cat(row_tensors, dim=row_dim)
    return row_tensors[0]                      # row-replicated


# ---------------------------------------------------------------------------
# PCC in fp64 with chunked accumulation (handles 13B-element tensors).
# ---------------------------------------------------------------------------


def _chunk_sum_f64(x: torch.Tensor, chunk: int = 1_000_000) -> float:
    total = 0.0
    flat = x.reshape(-1)
    for start in range(0, flat.numel(), chunk):
        total += flat[start : start + chunk].to(torch.float64).sum().item()
    return total


def compute_pcc_f64(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    n = a.numel()
    if n == 0:
        return float("nan")
    mean_a = _chunk_sum_f64(a) / n
    mean_b = _chunk_sum_f64(b) / n
    num = va = vb = 0.0
    fa = a.reshape(-1)
    fb = b.reshape(-1)
    chunk = 1_000_000
    for start in range(0, n, chunk):
        ac = fa[start : start + chunk].to(torch.float64) - mean_a
        bc = fb[start : start + chunk].to(torch.float64) - mean_b
        num += (ac * bc).sum().item()
        va += (ac * ac).sum().item()
        vb += (bc * bc).sum().item()
    denom = (va ** 0.5) * (vb ** 0.5)
    if denom == 0.0:
        return float("nan")
    return num / denom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _pcc_pair(a: torch.Tensor, b: torch.Tensor) -> float:
    """Wrapper that reshapes flat if shapes differ but numel matches."""
    if a.shape != b.shape:
        if a.numel() != b.numel():
            return float("nan")
        a = a.reshape(-1)
        b = b.reshape(-1)
    return compute_pcc_f64(a, b)


def _match_best_for_cpu(
    cpu_tensor: torch.Tensor,
    device_list: list[torch.Tensor],
    used: set[int],
) -> tuple[int | None, torch.Tensor | None, float]:
    """Pick the unused device output whose reconstruction produces the best
    PCC against cpu_tensor. Needed because shapes alone can't disambiguate
    between kv_keys vs kv_values (both (64, 8, 128, 64))."""
    target = tuple(cpu_tensor.shape)
    best_idx, best_rec, best_pcc = None, None, -2.0
    for idx, d in enumerate(device_list):
        if idx in used:
            continue
        if d.shape[0] != MESH_ROWS * MESH_COLS or len(d.shape) - 1 != len(target):
            continue
        rec = _reconstruct_from_mesh(d, target)
        if rec.numel() != cpu_tensor.numel():
            continue
        pcc = _pcc_pair(rec, cpu_tensor)
        if pcc > best_pcc:
            best_idx, best_rec, best_pcc = idx, rec, pcc
    return best_idx, best_rec, best_pcc


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <generated_dir>", file=sys.stderr)
        sys.exit(2)
    gen = Path(sys.argv[1]).resolve()

    print("Loading CPU reference...")
    cpu_data = torch.load(gen / "cpu_reference_outputs.pt", weights_only=True, map_location="cpu")
    cpu_tensors = cpu_data["tensors"]
    cpu_names = cpu_data["names"]
    # Only compare to the CPU reference entries whose name identifies them as
    # a real model output — logits and the single-layer KV cache. The full
    # StaticCache contained 36 layers because HF's default config has 36
    # layers; layers 1..35 are zero scratch (model only has 1 actual layer).
    keep = [
        (i, n) for i, n in enumerate(cpu_names)
        if n == "logits" or n == "kv_keys_layer0" or n == "kv_values_layer0"
    ]
    print(f"CPU reference: {len(cpu_tensors)} tensors; comparing {len(keep)} (logits + layer-0 KV).")

    print(f"Loading device outputs...")
    device_list = torch.load(gen / "device_outputs.pt", weights_only=True, map_location="cpu")
    print(f"Loading bypass outputs...")
    bypass_list = torch.load(gen / "bypass_outputs.pt", weights_only=True, map_location="cpu")

    print("\n=== PCC vs CPU reference (bf16, non-compiled) ===")
    print(f"{'cpu entry':<22} {'shape':<28} {'dev idx→PCC':<22} {'byp idx→PCC':<22}")
    print("-" * 94)
    used_dev: set[int] = set()
    used_byp: set[int] = set()
    matched = {}  # name -> (cpu_t, dev_rec, byp_rec)
    for i, name in keep:
        cpu_t = cpu_tensors[i]
        d_idx, d_rec, d_pcc = _match_best_for_cpu(cpu_t, device_list, used_dev)
        b_idx, b_rec, b_pcc = _match_best_for_cpu(cpu_t, bypass_list, used_byp)
        if d_idx is not None:
            used_dev.add(d_idx)
        if b_idx is not None:
            used_byp.add(b_idx)
        d_str = f"[{d_idx}] → {d_pcc:+.6f}" if d_idx is not None else "NO MATCH"
        b_str = f"[{b_idx}] → {b_pcc:+.6f}" if b_idx is not None else "NO MATCH"
        print(f"{name:<22} {str(tuple(cpu_t.shape)):<28} {d_str:<22} {b_str:<22}")
        if d_rec is not None and b_rec is not None:
            matched[name] = (cpu_t, d_rec, b_rec)

    # Extra comparisons matching progressively more of what pytest does:
    #   (a) last-token logits, all batch samples: logits[:, -1]
    #   (b) pytest exact: output_logits[0][0] = first batch, last token only
    if "logits" in matched:
        cpu_t, d_rec, b_rec = matched["logits"]
        print("\n=== Progressively narrower logit slices vs CPU reference ===")
        print(f"{'slice':<38} {'shape':<20} {'dev vs CPU':<16} {'bypass vs CPU':<16}")
        print("-" * 95)

        # (a) all batch, last token
        cpu_a = cpu_t[:, -1]
        d_a = d_rec[:, -1]
        b_a = b_rec[:, -1]
        print(f"{'all batch, last token (batch,vocab)':<38} "
              f"{str(tuple(cpu_a.shape)):<20} "
              f"{compute_pcc_f64(d_a, cpu_a):+.6f}{'':<6} "
              f"{compute_pcc_f64(b_a, cpu_a):+.6f}")

        # (b) pytest exact: first batch sample only, last token
        cpu_b = cpu_t[0, -1]
        d_b = d_rec[0, -1]
        b_b = b_rec[0, -1]
        print(f"{'pytest exact: batch[0], last token':<38} "
              f"{str(tuple(cpu_b.shape)):<20} "
              f"{compute_pcc_f64(d_b, cpu_b):+.6f}{'':<6} "
              f"{compute_pcc_f64(b_b, cpu_b):+.6f}")


if __name__ == "__main__":
    main()
