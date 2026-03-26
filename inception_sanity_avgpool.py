#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Inception v4 AvgPool2d NaN/inf repro
======================================
Minimal single-op test that reproduces the PCC failure in Inception v4.

Root cause isolation:
  features[0:6]  (stem + Mixed3a/4a/5a)  PCC=0.9999  PASS
  features[0:7]  (+InceptionA[6])        TT has inf  FAIL

  InceptionA.branch3:
    AvgPool2d(kernel_size=3, stride=1, padding=1)   ← FAILING OP
    → ConvNormAct(384→96, 1×1)

  AvgPool2d alone:
    CPU output: min=0.00  max=9.33  mean=0.735  std=1.093  clean
    TT  output: nan=True  inf=True  — completely broken

This test runs just AvgPool2d(kernel_size=3, stride=1, padding=1) with a
realistic input tensor (shape [1, 384, 35, 35], positive values matching
the actual block input statistics).

Usage:
    cd /proj_sw/user_dev/ctr-lelanchelian/latest_build/tt-xla
    source venv/activate
    python inception_sanity_avgpool.py
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch_xla
import torch_plugin_tt  # noqa: F401

PCC_THRESHOLD = 0.99


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.detach().float().cpu().flatten()
    b_f = b.detach().float().cpu().flatten()
    if torch.allclose(a_f, b_f, rtol=1e-5, atol=1e-5):
        return 1.0
    va = a_f - a_f.mean()
    vb = b_f - b_f.mean()
    denom = va.norm() * vb.norm()
    return float("nan") if float(denom) == 0.0 else float((va @ vb) / denom)


def run_on_tt(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    xla_device = torch_xla.device()
    m = copy.deepcopy(model).eval().to(xla_device)
    inp = inputs.to(xla_device)
    compiled = torch.compile(m, backend="tt", options={})
    with torch.no_grad():
        out = compiled(inp)
    torch_xla.sync()
    return out.cpu()


def main():
    print("=" * 64)
    print("Inception v4 — AvgPool2d(3, s=1, p=1) sanity test")
    print("=" * 64)

    # The exact op from InceptionA.branch3
    op = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    # Input matching actual block statistics:
    #   shape [1, 384, 35, 35], non-negative (post-ReLU), mean≈0.74, std≈1.21
    torch.manual_seed(0)
    inp = torch.abs(torch.randn(1, 384, 35, 35)) * 1.21

    print(f"\nOp     : AvgPool2d(kernel_size=3, stride=1, padding=1)")
    print(f"Input  : shape={list(inp.shape)}  min={inp.min():.4f}  max={inp.max():.4f}"
          f"  mean={inp.mean():.4f}  std={inp.std():.4f}")

    # CPU reference
    with torch.no_grad():
        cpu_out = op(inp)
    print(f"\nCPU out: shape={list(cpu_out.shape)}  min={cpu_out.min():.4f}  max={cpu_out.max():.4f}"
          f"  mean={cpu_out.mean():.4f}  std={cpu_out.std():.4f}"
          f"  nan={bool(torch.isnan(cpu_out).any())}  inf={bool(torch.isinf(cpu_out).any())}")

    # TT run
    tt_out = run_on_tt(op, inp)
    print(f"TT  out: shape={list(tt_out.shape)}  min={tt_out.min():.4f}  max={tt_out.max():.4f}"
          f"  mean={tt_out.mean():.4f}  std={tt_out.std():.4f}"
          f"  nan={bool(torch.isnan(tt_out).any())}  inf={bool(torch.isinf(tt_out).any())}")

    pcc = compute_pcc(cpu_out, tt_out)
    pcc_str = f"{pcc:.6f}"
    status = "PASS" if not (pcc != pcc) and pcc >= PCC_THRESHOLD else "FAIL"

    print(f"\nPCC    : {pcc_str}  (threshold={PCC_THRESHOLD})")
    print(f"Result : {status}")
    print("=" * 64)

    if status == "FAIL":
        print("\nREPRO CONFIRMED — TT produces nan/inf for AvgPool2d(3,s=1,p=1)")
        print("This is the root cause of Inception v4 PCC drop (pcc=0.014 in full model).")
        print("\nOp location in model:")
        print("  timm inception_v4")
        print("  └── features[6]  InceptionA")
        print("       └── branch3[0]  AvgPool2d(kernel_size=3, stride=1, padding=1)")
        print("\nSame class of issue as DenseNet121 AvgPool2d repro.")


if __name__ == "__main__":
    main()
