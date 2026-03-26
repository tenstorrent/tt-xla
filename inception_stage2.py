#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Inception v4 Stage 2 drill-down: features[6..9] (InceptionA × 4)

Stage 1 established:
  features[0:6]  (stem + Mixed3a/4a/5a)   PCC = 0.9999  PASS
  features[0:10] (+ InceptionA × 4)       PCC = nan     FAIL

Each InceptionA has:
  branch3: AvgPool2d(kernel_size=3, stride=1, padding=1) → ConvNormAct

This script tests features[0:7], [0:8], [0:9], [0:10] to find
which InceptionA is the first to fail.
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


def test_slice(features, end_idx: int, inputs: torch.Tensor, label: str) -> float:
    m = nn.Sequential(*list(features.children())[:end_idx]).eval()
    with torch.no_grad():
        cpu_out = m(inputs)
    print(f"  CPU: {tensor_stats(cpu_out)}")

    try:
        tt_out = run_tt(m, inputs)
        print(f"  TT : {tensor_stats(tt_out)}")
        pcc = compute_pcc(cpu_out, tt_out)
    except Exception:
        print(f"  TT run failed:\n{traceback.format_exc()}")
        pcc = None

    status = "PASS" if pcc is not None and not (pcc != pcc) and pcc >= 0.99 else "FAIL"
    pcc_str = f"{pcc:.6f}" if pcc is not None else "N/A"
    print(f"  PCC = {pcc_str}  [{status}]\n")
    return pcc


def main():
    import timm
    print("Loading inception_v4...")
    model = timm.create_model("inception_v4", pretrained=True).eval()
    inputs = torch.randn(1, 3, 299, 299)
    features = model.features
    feature_list = list(features.children())

    print("=" * 64)
    print("STAGE 2: InceptionA block drill-down (features[6..9])")
    print("  Stage 1 showed features[0:6] PASS, features[0:10] FAIL")
    print("=" * 64)

    for end_idx in range(7, 11):  # 7, 8, 9, 10
        block = feature_list[end_idx - 1]
        label = f"features[0:{end_idx}]  (+{type(block).__name__} at [{end_idx-1}])"
        print(f"\n{'─'*64}")
        print(f"{label}")
        print(f"{'─'*64}")
        pcc = test_slice(features, end_idx, inputs, label)
        if pcc is None or (pcc != pcc) or pcc < 0.99:
            print(f"*** FIRST FAILING BLOCK: features[{end_idx-1}] = {type(block).__name__} ***")
            print(f"  Block detail: {block}")
            # Now drill into individual branches of this block
            print(f"\n{'='*64}")
            print(f"STAGE 3: Drilling into {type(block).__name__} branches")
            print(f"{'='*64}")
            drill_into_inception_block(features, end_idx - 1, inputs)
            break

    print("\nDone.")


def drill_into_inception_block(features, block_idx: int, inputs: torch.Tensor):
    """Test each branch of the failing InceptionA block independently on the same input."""
    feature_list = list(features.children())
    block = feature_list[block_idx]

    # Get the input to this block (run features[0:block_idx] on CPU)
    prefix = nn.Sequential(*feature_list[:block_idx]).eval()
    with torch.no_grad():
        block_input = prefix(inputs)
    print(f"  Block input: {tensor_stats(block_input)}\n")

    # Test full block
    print(f"  Testing full {type(block).__name__}:")
    pcc = _test_submodule(block, block_input, "full_block")
    print(f"  Full block PCC = {pcc:.6f if pcc is not None else 'N/A'}\n")

    # Test each branch independently
    for branch_name, branch_mod in block.named_children():
        print(f"  Testing branch '{branch_name}':")
        pcc = _test_submodule(branch_mod, block_input, branch_name)
        pcc_str = f"{pcc:.6f}" if pcc is not None else "N/A"
        status = "PASS" if pcc is not None and not (pcc != pcc) and pcc >= 0.99 else "FAIL"
        print(f"  Branch '{branch_name}' PCC = {pcc_str}  [{status}]\n")


def _test_submodule(mod: nn.Module, inputs: torch.Tensor, label: str) -> float:
    mod = mod.eval()
    with torch.no_grad():
        cpu_out = mod(inputs)
    print(f"    CPU: {tensor_stats(cpu_out)}")
    try:
        tt_out = run_tt(mod, inputs)
        print(f"    TT : {tensor_stats(tt_out)}")
        pcc = compute_pcc(cpu_out, tt_out)
    except Exception:
        print(f"    TT run failed:\n{traceback.format_exc()}")
        pcc = None
    return pcc


if __name__ == "__main__":
    main()
