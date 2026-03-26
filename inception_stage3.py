#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Inception v4 Stage 3: drill into InceptionA (features[6]) branches.

Stage 2 established:
  features[0:7]  (+InceptionA at [6])  TT output has inf → PCC=nan  FAIL

InceptionA branches:
  branch0: ConvNormAct(384→96, 1×1)
  branch1: ConvNormAct(384→64,1×1) → ConvNormAct(64→96,3×3)
  branch2: ConvNormAct(384→64,1×1) → ConvNormAct(64→96,3×3) → ConvNormAct(96→96,3×3)
  branch3: AvgPool2d(3,s=1,p=1) → ConvNormAct(384→96,1×1)

This script tests each branch independently using the same block input tensor.
"""

import copy
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch_xla
import torch_plugin_tt  # noqa: F401

LOG_DIR = PROJECT_ROOT / "inception"
LOG_DIR.mkdir(exist_ok=True)


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.detach().float().cpu().flatten()
    b_f = b.detach().float().cpu().flatten()
    if torch.allclose(a_f, b_f, rtol=1e-5, atol=1e-5):
        return 1.0
    va = a_f - a_f.mean()
    vb = b_f - b_f.mean()
    denom = va.norm() * vb.norm()
    return float("nan") if float(denom) == 0.0 else float((va @ vb) / denom)


def tensor_stats(t: torch.Tensor) -> str:
    f = t.detach().float().cpu()
    return (f"shape={list(t.shape)}  min={f.min():.4f}  max={f.max():.4f}  "
            f"mean={f.mean():.6f}  std={f.std():.6f}  "
            f"nan={bool(torch.isnan(f).any())}  inf={bool(torch.isinf(f).any())}")


def run_tt(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    xla_device = torch_xla.device()
    m = copy.deepcopy(model).eval().to(xla_device)
    inp = inputs.to(xla_device)
    compiled = torch.compile(m, backend="tt", options={})
    with torch.no_grad():
        out = compiled(inp)
    torch_xla.sync()
    return out.cpu()


def test_module(mod: nn.Module, inputs: torch.Tensor, label: str) -> None:
    mod = mod.eval()
    with torch.no_grad():
        cpu_out = mod(inputs)
    print(f"  CPU: {tensor_stats(cpu_out)}")
    try:
        tt_out = run_tt(mod, inputs)
        print(f"  TT : {tensor_stats(tt_out)}")
        pcc = compute_pcc(cpu_out, tt_out)
        pcc_str = f"{pcc:.6f}"
        status = "PASS" if not (pcc != pcc) and pcc >= 0.99 else "FAIL"
    except Exception:
        print(f"  TT run failed:\n{traceback.format_exc()}")
        pcc_str = "ERROR"
        status = "FAIL"
    print(f"  PCC = {pcc_str}  [{status}]")


def main():
    import timm
    print("Loading inception_v4...")
    model = timm.create_model("inception_v4", pretrained=True).eval()
    inputs = torch.randn(1, 3, 299, 299)
    features = model.features
    feature_list = list(features.children())

    # Get input to features[6] (InceptionA) by running features[0:6] on CPU
    prefix = nn.Sequential(*feature_list[:6]).eval()
    with torch.no_grad():
        block_input = prefix(inputs)
    print(f"\nBlock input to InceptionA (features[6]): {tensor_stats(block_input)}\n")

    inception_a = feature_list[6]  # InceptionA

    print("=" * 64)
    print("STAGE 3: InceptionA branch drill-down")
    print("  TT output of full InceptionA has inf → PCC=nan")
    print("=" * 64)

    # Test each branch individually
    for branch_name, branch_mod in inception_a.named_children():
        print(f"\n{'─'*64}")
        print(f"Branch '{branch_name}': {branch_mod}")
        print(f"{'─'*64}")
        test_module(branch_mod, block_input, branch_name)

    # Also test full block
    print(f"\n{'─'*64}")
    print("Full InceptionA block (concat of all branches):")
    print(f"{'─'*64}")
    test_module(inception_a, block_input, "full_InceptionA")

    # If branch3 (AvgPool + Conv) is suspect, drill further
    print(f"\n{'='*64}")
    print("STAGE 4: branch3 sub-op drill-down")
    print("  branch3: AvgPool2d(3,s=1,p=1) → ConvNormAct(384→96,1×1)")
    print(f"{'='*64}")

    branch3 = inception_a.branch3

    # Op 1: just AvgPool2d
    print(f"\n  Op 1 — AvgPool2d alone:")
    avgpool = branch3[0]
    print(f"         {avgpool}")
    test_module(avgpool, block_input, "AvgPool2d")

    # Op 2: AvgPool2d → ConvNormAct
    print(f"\n  Op 2 — AvgPool2d → ConvNormAct(384→96,1×1):")
    test_module(branch3, block_input, "branch3_full")

    print("\nDone.")


if __name__ == "__main__":
    main()
