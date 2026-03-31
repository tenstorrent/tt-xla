#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Verify CenterPoint TT output against real CPU ground truth.

Steps:
  1. Load pre-saved RPN+CenterHead weights (extracted from real det3d model)
  2. Load weights into standalone CenterPointRPNHead
  3. Load saved BEV image (from extract_centerpoint_ground_truth.py)
  4. CPU sanity check: standalone (real weights) vs saved GT
  5. Run on TT device via torch.compile("tt")
  6. Compare TT output against saved CPU ground truth

Pre-requisites (run once with venv_centerpoint):
    PYTHONPATH=... python scripts/extract_centerpoint_ground_truth.py

Run from tt-xla root with tt-xla venv:
    source venv/bin/activate
    python scripts/verify_centerpoint_vs_cpu_gt.py
"""

import sys, os
import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, ".")
sys.path.insert(0, "third_party")

WEIGHTS_FILE = "tests/torch/graphs/centerpoint_rpn_head_weights.pt"
GT_BEV       = "tests/torch/graphs/centerpoint_bev_image_bf16.pt"
GT_PREDS     = "tests/torch/graphs/centerpoint_raw_preds_bf16.pt"


# ── PCC helper ────────────────────────────────────────────────────────────────

def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    if torch.allclose(a, b, atol=1e-3): return 1.0
    # Near-constant signal: range < 2 bfloat16 ULPs at the signal magnitude.
    # This happens for untrained hm heads (all values ≈ -2.19 init bias).
    # PCC is numerically meaningless here; use allclose with 2 ULP tolerance.
    sig_range = (b.max() - b.min()).item()
    if sig_range < 3.5e-2 and torch.allclose(a, b, atol=3.5e-2):
        return 1.0
    va, vb = a - a.mean(), b - b.mean()
    d = va.norm() * vb.norm()
    return float((va @ vb) / d) if d > 0 else float("nan")


# ── Step 1: load pre-saved weights ───────────────────────────────────────────

print("=" * 65)
print("Step 1 — load pre-saved RPN + CenterHead weights")
print("=" * 65)

assert os.path.exists(WEIGHTS_FILE), (
    f"Missing: {WEIGHTS_FILE}\n"
    "  Run scripts/extract_centerpoint_ground_truth.py first (in venv_centerpoint)"
)

saved = torch.load(WEIGHTS_FILE, weights_only=True)
rpn_sd  = saved["rpn"]
head_sd = saved["head"]
print(f"  Loaded {len(rpn_sd)} RPN params, {len(head_sd)} head params")


# ── Step 2: load weights into standalone model ────────────────────────────────

print("\nStep 2 — load weights into standalone CenterPointRPNHead")

from tt_forge_models.centerpoint.pytorch.src.model import CenterPointRPNHead

standalone = CenterPointRPNHead().eval()

rpn_missing, rpn_unexpected = standalone.rpn.load_state_dict(rpn_sd, strict=True)
print(f"  RPN load: missing={rpn_missing}, unexpected={rpn_unexpected}")

head_missing, head_unexpected = standalone.head.load_state_dict(head_sd, strict=True)
print(f"  Head load: missing={head_missing}, unexpected={head_unexpected}")


# ── Step 3: load saved BEV image + CPU ground truth ──────────────────────────

print("\nStep 3 — load saved BEV image and CPU ground truth")

assert os.path.exists(GT_BEV),   f"Missing: {GT_BEV}   — run extract_centerpoint_ground_truth.py first"
assert os.path.exists(GT_PREDS), f"Missing: {GT_PREDS}  — run extract_centerpoint_ground_truth.py first"

bev_image = torch.load(GT_BEV,   weights_only=True)   # (1, 64, 512, 512) bf16
gt_preds  = torch.load(GT_PREDS, weights_only=True)   # list of 6 task dicts, bf16
print(f"  BEV image: {list(bev_image.shape)} {bev_image.dtype}")
print(f"  GT tasks:  {len(gt_preds)}")


# ── Step 4: CPU sanity check (standalone with real weights vs GT) ─────────────

print("\nStep 4 — CPU sanity check: standalone model (real weights) vs GT")

standalone_bf16 = standalone.to(torch.bfloat16)
with torch.no_grad():
    cpu_preds = standalone_bf16(bev_image)

all_pcc = []
for t_idx, (cpu_task, gt_task) in enumerate(zip(cpu_preds, gt_preds)):
    for head_name in gt_task:
        p = pcc(cpu_task[head_name], gt_task[head_name])
        all_pcc.append(p)
        if abs(p - 1.0) > 1e-4:
            print(f"  task[{t_idx}].{head_name}: PCC={p:.6f}  <- mismatch!")

min_pcc = min(all_pcc)
print(f"  CPU standalone vs GT: min PCC = {min_pcc:.6f}  ({'PASS' if min_pcc > 0.99 else 'FAIL'})")
if min_pcc < 0.99:
    print("  ERROR: weight transfer failed — standalone CPU != det3d CPU")
    sys.exit(1)


# ── Step 5: TT inference ──────────────────────────────────────────────────────

print("\nStep 5 — TT inference")

tt_device = xm.xla_device()
print(f"  TT device: {tt_device}")

standalone_tt = standalone.to(tt_device)
bev_tt = bev_image.to(tt_device)

compiled = torch.compile(standalone_tt, backend="tt", fullgraph=True)
with torch.no_grad():
    tt_preds_raw = compiled(bev_tt)
xm.mark_step()

# bring back to CPU
tt_preds = []
for task_dict in tt_preds_raw:
    tt_preds.append({k: v.detach().cpu().float() for k, v in task_dict.items()})


# ── Step 6: compare TT vs CPU ground truth ───────────────────────────────────

print("\nStep 6 — TT vs CPU ground truth comparison")
print(f"  {'Task/Head':<30} {'PCC':>10}  {'MaxAbsErr':>12}")
print("  " + "-" * 56)

all_pcc_tt = []
for t_idx, (tt_task, gt_task) in enumerate(zip(tt_preds, gt_preds)):
    for head_name in sorted(gt_task.keys()):
        p = pcc(tt_task[head_name], gt_task[head_name].float())
        err = (tt_task[head_name] - gt_task[head_name].float()).abs().max().item()
        all_pcc_tt.append(p)
        label = f"task[{t_idx}].{head_name}"
        print(f"  {label:<30} {p:>10.6f}  {err:>12.4e}")

min_pcc_tt = min(all_pcc_tt)
print()
print("=" * 65)
status = "PASS" if min_pcc_tt >= 0.99 else "FAIL"
print(f"  Result : {status}")
print(f"  Min PCC (TT vs real CPU GT) : {min_pcc_tt:.6f}  (threshold >= 0.99)")
print("=" * 65)
