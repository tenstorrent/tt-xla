# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Compare TTNN vs CPU intermediate tensors saved by debug_nan.py.

Loads all .pt files from /tmp/zt_debug/{tt,cpu}/ and prints:
  - shape, mean, std, NaN/inf count for each
  - PCC between TTNN and CPU for matching files

Usage:
    python compare_tensors.py
"""

import os
import glob
import torch

DUMP_TT  = "/tmp/zt_debug/tt"
DUMP_CPU = "/tmp/zt_debug/cpu"


def stats(t, label=""):
    valid = t[~torch.isnan(t) & ~torch.isinf(t)]
    mean = float(valid.mean()) if valid.numel() > 0 else float("nan")
    std  = float(valid.std())  if valid.numel() > 1 else 0.0
    nan  = int(torch.isnan(t).sum())
    inf  = int(torch.isinf(t).sum())
    return f"{label:40s}  shape={str(tuple(t.shape)):25s}  mean={mean:9.4f}  std={std:8.4f}  nan={nan}  inf={inf}"


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    # Match sizes by taking the min (CPU might have different shape from sharded TTNN)
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    if torch.isnan(a).any() or torch.isnan(b).any():
        return float("nan")
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    print(f"\n{'='*80}")
    print(f"Comparing TTNN vs CPU tensors")
    print(f"  TT  dumps: {DUMP_TT}")
    print(f"  CPU dumps: {DUMP_CPU}")
    print(f"{'='*80}\n")

    # Collect all tensor names (strip tt_/cpu_ prefix and .pt suffix)
    tt_files  = {os.path.basename(f)[3:-3]: f  # strip "tt_" and ".pt"
                 for f in sorted(glob.glob(os.path.join(DUMP_TT,  "tt_*.pt")))}
    cpu_files = {os.path.basename(f)[4:-3]: f  # strip "cpu_" and ".pt"
                 for f in sorted(glob.glob(os.path.join(DUMP_CPU, "cpu_*.pt")))}

    all_names = sorted(set(tt_files) | set(cpu_files))

    if not all_names:
        print("No tensors found. Run debug_nan.py first.")
        return

    fmt = f"{'Name':40s}  {'Shape':25s}  {'Mean':>9}  {'Std':>8}  Nan  Inf"
    print(f"{'':40s}  {'':25s}  {'TTNN vs CPU':^30}")
    print("-" * 120)

    nan_first_introduced = None
    for name in all_names:
        has_tt  = name in tt_files
        has_cpu = name in cpu_files

        tt_t  = torch.load(tt_files[name],  weights_only=True).float() if has_tt  else None
        cpu_t = torch.load(cpu_files[name], weights_only=True).float() if has_cpu else None

        tt_str  = stats(tt_t,  "TT ") if tt_t  is not None else "    (missing)"
        cpu_str = stats(cpu_t, "CPU") if cpu_t is not None else "    (missing)"

        tt_has_nan  = has_tt  and bool(torch.isnan(tt_t).any())
        cpu_has_nan = has_cpu and bool(torch.isnan(cpu_t).any())

        if has_tt and has_cpu:
            p = pcc(tt_t, cpu_t)
            pcc_str = f"PCC={p:.4f}"
        else:
            pcc_str = ""

        marker = ""
        if tt_has_nan and not cpu_has_nan:
            marker = "  <<<< TTNN NaN (CPU OK)"
            if nan_first_introduced is None:
                nan_first_introduced = name
        elif tt_has_nan and cpu_has_nan:
            marker = "  (both NaN)"

        print(f"\n[{name}]")
        print(f"  {tt_str}")
        print(f"  {cpu_str}")
        if pcc_str:
            print(f"  {pcc_str}{marker}")
        elif marker:
            print(f"  {marker}")

    print("\n" + "="*80)
    if nan_first_introduced:
        print(f"  *** First TTNN NaN introduced at: {nan_first_introduced} ***")
    else:
        print("  No TTNN-only NaN found in saved tensors.")
    print("="*80)


if __name__ == "__main__":
    main()
