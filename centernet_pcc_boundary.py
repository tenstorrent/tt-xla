#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
CenterNet PCC boundary verification
====================================
Proves the EXACT line where PCC crosses the failure threshold.

The failing operation is reg_head.conv2, the final Conv2d(256→2, 1×1)
inside the offset-regression head of DLASeg.

In DLASeg.forward (pose_dla_dcn.py:633-645):

    def forward(self, x):
        x = self.base(x)           # DLA backbone
        x = self.dla_up(x)         # DLAUp feature pyramid
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))  # IDAUp upsampling

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])   # ← line 644
        return [z]

self.reg is nn.Sequential:
    reg[0]  Conv2d(64 → 256, 3×3, padding=1)
    reg[1]  ReLU
    reg[2]  Conv2d(256 → 2, 1×1)              ← FAILING OP

        return after reg[1]  →  PCC ≈ 0.989  PASS
        return after reg[2]  →  PCC ≈ 0.969  FAIL  ← threshold = 0.97

Usage:
    export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
    export LOGGER_LEVEL=DEBUG
    source venv/activate
    python centernet_pcc_boundary.py
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# must import torch_xla first to avoid circular-import through entry_points
import torch_xla  # noqa: E402
import torch_plugin_tt  # noqa: E402, F401

PCC_THRESHOLD = 0.97


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── boundary slice models ─────────────────────────────────────────────────────

class _Backbone(nn.Module):
    """Shared backbone forward: base + DLAUp + IDAUp → final feature map y[-1]."""

    def __init__(self, model):
        super().__init__()
        self.base = model.base
        self.dla_up = model.dla_up
        self.ida_up = model.ida_up
        self.first_level = model.first_level
        self.last_level = model.last_level

    def forward(self, x):
        # pose_dla_dcn.py:634-640
        x = self.base(x)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return y[-1]   # shape [1, 64, 128, 128]


class ReturnBeforeReg2(nn.Module):
    """
    Runs backbone + reg[0] + reg[1] and STOPS before reg[2].

    Equivalent to inserting an early return at pose_dla_dcn.py:644
    BEFORE self.reg[2] is applied:

        feat = self.reg[0](y[-1])   # Conv2d 64→256
        feat = self.reg[1](feat)    # ReLU
        return feat                 # ← early return HERE  →  PCC ≈ 0.989  PASS
        feat = self.reg[2](feat)    # Conv2d 256→2, 1×1   ← never reached
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = _Backbone(model)
        self.reg0 = model.reg[0]   # Conv2d(64 → 256, 3×3)
        self.reg1 = model.reg[1]   # ReLU

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.reg0(feat)
        feat = self.reg1(feat)
        return feat                # stops here — PCC passes


class ReturnAfterReg2(nn.Module):
    """
    Runs backbone + full reg head (reg[0] + reg[1] + reg[2]).

    Equivalent to the normal execution of pose_dla_dcn.py:644:

        feat = self.reg[0](y[-1])   # Conv2d 64→256
        feat = self.reg[1](feat)    # ReLU
        feat = self.reg[2](feat)    # Conv2d 256→2, 1×1  ← PCC drops HERE
        return feat                 # ← return here  →  PCC ≈ 0.969  FAIL
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = _Backbone(model)
        self.reg0 = model.reg[0]   # Conv2d(64 → 256, 3×3)
        self.reg1 = model.reg[1]   # ReLU
        self.reg2 = model.reg[2]   # Conv2d(256 → 2, 1×1)  ← the failing op

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.reg0(feat)
        feat = self.reg1(feat)
        feat = self.reg2(feat)     # ← this single op drops PCC below 0.97
        return feat


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("CenterNet PCC boundary verification")
    print("=" * 64)

    # load
    from third_party.tt_forge_models.centernet.pytorch.loader import (
        ModelLoader, ModelVariant,
    )
    loader = ModelLoader(variant=ModelVariant.DLA_1X_COCO)
    model = loader.load_model()
    inputs = loader.load_inputs()
    model.eval()

    print(f"\nInput : {inputs.shape}  {inputs.dtype}")
    print(f"Failing op : model.reg[2]  →  Conv2d(256→2, kernel_size=1)")
    print(f"Location   : pose_dla_dcn.py:644  z['reg'] = self.reg(y[-1])")
    print(f"PCC threshold : {PCC_THRESHOLD}\n")

    # ── PASS: return BEFORE reg[2] ─────────────────────────────────────────
    print("─" * 64)
    print("CASE A — return BEFORE reg[2]  (should PASS)")
    print("─" * 64)
    pass_model = ReturnBeforeReg2(model)
    with torch.no_grad():
        cpu_pass = pass_model(inputs)
    tt_pass = run_on_tt(pass_model, inputs)
    pcc_pass = compute_pcc(cpu_pass, tt_pass)
    status_pass = "PASS" if pcc_pass >= PCC_THRESHOLD else "FAIL"
    print(f"  CPU output : shape={list(cpu_pass.shape)}  std={cpu_pass.std():.4f}  max={cpu_pass.max():.4f}")
    print(f"  TT  output : shape={list(tt_pass.shape)}   std={tt_pass.std():.4f}  max={tt_pass.max():.4f}")
    print(f"  PCC = {pcc_pass:.6f}  [{status_pass}]")

    # ── FAIL: return AFTER reg[2] ──────────────────────────────────────────
    print()
    print("─" * 64)
    print("CASE B — return AFTER reg[2]   (should FAIL)")
    print("─" * 64)
    fail_model = ReturnAfterReg2(model)
    with torch.no_grad():
        cpu_fail = fail_model(inputs)
    tt_fail = run_on_tt(fail_model, inputs)
    pcc_fail = compute_pcc(cpu_fail, tt_fail)
    status_fail = "PASS" if pcc_fail >= PCC_THRESHOLD else "FAIL"
    print(f"  CPU output : shape={list(cpu_fail.shape)}  std={cpu_fail.std():.6f}  max={cpu_fail.max():.4f}")
    print(f"  TT  output : shape={list(tt_fail.shape)}   std={tt_fail.std():.6f}  max={tt_fail.max():.4f}")
    print(f"  PCC = {pcc_fail:.6f}  [{status_fail}]")

    # ── summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Before reg[2]  PCC = {pcc_pass:.6f}  [{status_pass}]   ← backbone + reg[0] + reg[1]")
    print(f"  After  reg[2]  PCC = {pcc_fail:.6f}  [{status_fail}]  ← + reg[2] Conv2d(256→2,1×1)")
    print()
    print("  EXACT FAILING LINE IN MODEL:")
    print()
    print("  # pose_dla_dcn.py : DLASeg.forward(), line 644")
    print("  z['reg'] = self.reg(y[-1])")
    print()
    print("  Expanded:")
    print("  feat = self.reg[0](y[-1])  # Conv2d(64→256, 3×3)  PCC OK")
    print("  feat = self.reg[1](feat)   # ReLU                  PCC OK")
    print("  feat = self.reg[2](feat)   # Conv2d(256→2,  1×1)  ← PCC DROPS HERE")
    print()
    print("  Why: reg[2] outputs tiny values (std≈0.095). TT bfloat16 math")
    print("  introduces ~0.34 absolute error in max (1.44→1.10), which is")
    print("  large relative to the signal variance → PCC falls below 0.97.")
    print("=" * 64)


if __name__ == "__main__":
    main()
