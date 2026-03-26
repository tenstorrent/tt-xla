#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Inception v4 PCC Debug Script
==============================
Systematically isolates the exact operation causing PCC drop by progressively
slicing the Inception v4 (timm) model.

Model forward path:
  input (1, 3, 299, 299)
    → features[0-2]   ConvNormAct × 3  (stem)
    → features[3]     Mixed3a  (MaxPool2d + Conv3×3)
    → features[4]     Mixed4a
    → features[5]     Mixed5a
    → features[6-9]   InceptionA × 4   (each has AvgPool2d in branch3)
    → features[10]    ReductionA
    → features[11-17] InceptionB × 7   (each has AvgPool2d in branch3)
    → features[18]    ReductionB
    → features[19-21] InceptionC × 3   (each has AvgPool2d in branch3)
    → global_pool     SelectAdaptivePool2d (adaptive avg pool → flatten)
    → head_drop       Dropout
    → last_linear     Linear(1536→1000)

Usage:
    cd /proj_sw/user_dev/ctr-lelanchelian/latest_build/tt-xla
    source venv/activate
    python inception_pcc_debug.py

Logs:  inception/block1.log ... inception/debug_summary.log
       inception/pcc_summary.json
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── register TT backend early ─────────────────────────────────────────────────
try:
    import torch_xla
    import torch_plugin_tt  # noqa: F401

    TT_AVAILABLE = True
except ImportError as _e:
    print(f"[WARNING] torch_plugin_tt not importable: {_e}  — TT runs will be skipped.")
    TT_AVAILABLE = False

# ── output directory ──────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "inception"
LOG_DIR.mkdir(exist_ok=True)

# ── logging helpers ───────────────────────────────────────────────────────────
_loggers: Dict[str, logging.Logger] = {}


def _make_logger(name: str, filename: str) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / filename, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    _loggers[name] = log
    return log


main_log = _make_logger("main", "debug_summary.log")

# ── PCC & tensor statistics ───────────────────────────────────────────────────


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.detach().float().cpu().flatten()
    b_f = b.detach().float().cpu().flatten()
    if torch.allclose(a_f, b_f, rtol=1e-5, atol=1e-5):
        return 1.0
    if a_f.numel() <= 1:
        return 0.0
    va = a_f - a_f.mean()
    vb = b_f - b_f.mean()
    denom = va.norm() * vb.norm()
    return float("nan") if float(denom) == 0.0 else float((va @ vb) / denom)


def tensor_stats(t: torch.Tensor) -> Dict:
    f = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(f.min()),
        "max": float(f.max()),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "has_nan": bool(torch.isnan(f).any()),
        "has_inf": bool(torch.isinf(f).any()),
    }


def log_stats(log, tag: str, t: torch.Tensor) -> None:
    s = tensor_stats(t)
    log.debug(
        f"  {tag:6s}  shape={s['shape']}  dtype={s['dtype']}  "
        f"min={s['min']:.4f}  max={s['max']:.4f}  "
        f"mean={s['mean']:.6f}  std={s['std']:.6f}  "
        f"nan={s['has_nan']}  inf={s['has_inf']}"
    )


# ── device helpers ────────────────────────────────────────────────────────────


def run_cpu(model: nn.Module, inputs: torch.Tensor) -> Any:
    model.eval()
    with torch.no_grad():
        return model(inputs)


def run_tt(model: nn.Module, inputs: torch.Tensor, log) -> Optional[torch.Tensor]:
    if not TT_AVAILABLE:
        log.warning("TT backend not available — skipping TT run.")
        return None
    import copy
    try:
        xla_device = torch_xla.device()
        model_xla = copy.deepcopy(model)
        model_xla.eval()
        model_xla = model_xla.to(xla_device)
        inputs_xla = inputs.to(xla_device)
        model_tt = torch.compile(model_xla, backend="tt", options={})
        with torch.no_grad():
            out = model_tt(inputs_xla)
        torch_xla.sync()

        def _to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.cpu()
            if isinstance(x, (list, tuple)):
                return type(x)(_to_cpu(i) for i in x)
            if isinstance(x, dict):
                return {k: _to_cpu(v) for k, v in x.items()}
            return x

        return _to_cpu(out)
    except Exception:
        log.error("TT execution failed:\n" + traceback.format_exc())
        return None


def run_slice(
    slice_model: nn.Module,
    inputs: torch.Tensor,
    block_name: str,
    log_file: str,
    threshold: float = 0.99,
) -> Dict:
    log = _make_logger(block_name, log_file)
    log.info("=" * 62)
    log.info(f"Block : {block_name}")
    log.info("=" * 62)

    slice_model.eval()

    try:
        cpu_out = run_cpu(slice_model, inputs)
        cpu_tensor = cpu_out if isinstance(cpu_out, torch.Tensor) else cpu_out[0]
        log.info("CPU output:")
        log_stats(log, "CPU", cpu_tensor)
        cpu_stats = tensor_stats(cpu_tensor)
    except Exception:
        log.error("CPU run failed:\n" + traceback.format_exc())
        return {"block": block_name, "pcc": None, "log_file": log_file,
                "cpu_stats": None, "tt_stats": None}

    tt_out_raw = run_tt(slice_model, inputs, log)
    if tt_out_raw is None:
        log.warning("TT run returned None — PCC not computed.")
        return {"block": block_name, "pcc": None, "log_file": log_file,
                "cpu_stats": cpu_stats, "tt_stats": None}

    try:
        tt_tensor = tt_out_raw if isinstance(tt_out_raw, torch.Tensor) else tt_out_raw[0]
        log.info("TT output:")
        log_stats(log, "TT", tt_tensor)
        tt_stats = tensor_stats(tt_tensor)
    except Exception:
        log.error("Failed to extract TT tensor:\n" + traceback.format_exc())
        return {"block": block_name, "pcc": None, "log_file": log_file,
                "cpu_stats": cpu_stats, "tt_stats": None}

    pcc = compute_pcc(cpu_tensor, tt_tensor)
    ok = isinstance(pcc, float) and not (pcc != pcc) and pcc >= threshold
    status = "OK  " if ok else "FAIL"
    log.info(f"PCC = {pcc:.6f}  [{status}]")
    return {
        "block": block_name,
        "pcc": pcc,
        "log_file": log_file,
        "cpu_stats": cpu_stats,
        "tt_stats": tt_stats,
        "threshold": threshold,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Slice model wrappers
# ══════════════════════════════════════════════════════════════════════════════

class FeaturesSlice(nn.Module):
    """Run features[0:end_idx] only."""
    def __init__(self, features, end_idx: int):
        super().__init__()
        self.layers = nn.Sequential(*list(features.children())[:end_idx])

    def forward(self, x):
        return self.layers(x)


class FeaturesWithPool(nn.Module):
    """Run full features + global_pool (no linear)."""
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.global_pool = model.global_pool

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return x


class FullModel(nn.Module):
    """Run the full Inception v4 model (no dropout for determinism)."""
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.global_pool = model.global_pool
        self.last_linear = model.last_linear

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.last_linear(x)
        return x


# ── Fine-grained InceptionA/B/C drilldown ────────────────────────────────────

class FeaturesSliceWithExtra(nn.Module):
    """Run features[0:base_idx] then one extra block."""
    def __init__(self, features, base_idx: int, extra_block: nn.Module):
        super().__init__()
        self.base = nn.Sequential(*list(features.children())[:base_idx])
        self.extra = extra_block

    def forward(self, x):
        return self.extra(self.base(x))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import timm

    main_log.info("Loading inception_v4 (pretrained)...")
    model = timm.create_model("inception_v4", pretrained=True)
    model.eval()

    # Standard Inception v4 input: 1 × 3 × 299 × 299
    inputs = torch.randn(1, 3, 299, 299)
    main_log.info(f"Input shape: {inputs.shape}")

    results: List[Dict] = []
    threshold = 0.99

    # ── Stage 1: coarse block sweep ───────────────────────────────────────────
    #
    # features indices:
    #   [0-2]  : ConvNormAct × 3  (stem)
    #   [3]    : Mixed3a
    #   [4]    : Mixed4a
    #   [5]    : Mixed5a
    #   [6-9]  : InceptionA × 4
    #   [10]   : ReductionA
    #   [11-17]: InceptionB × 7
    #   [18]   : ReductionB
    #   [19-21]: InceptionC × 3
    #
    stage1_slices = [
        (3,  "stem_3conv",        "block1.log"),
        (6,  "stem+Mixed3a-5a",   "block2.log"),
        (10, "stem+Mixed+IncA×4", "block3.log"),
        (11, "..+ReductionA",     "block4.log"),
        (18, "..+IncB×7",         "block5.log"),
        (19, "..+ReductionB",     "block6.log"),
        (22, "full_features",     "block7.log"),
    ]

    main_log.info("\n" + "=" * 62)
    main_log.info("STAGE 1 — coarse sweep over feature blocks")
    main_log.info("=" * 62)

    first_fail_idx = None
    for end_idx, name, log_file in stage1_slices:
        m = FeaturesSlice(model.features, end_idx)
        r = run_slice(m, inputs, name, log_file, threshold)
        results.append(r)
        pcc_str = f"{r['pcc']:.6f}" if r['pcc'] is not None else "N/A"
        ok = r['pcc'] is not None and r['pcc'] >= threshold
        main_log.info(f"  {name:30s}  PCC={pcc_str}  {'OK' if ok else 'FAIL'}")
        if not ok and first_fail_idx is None:
            first_fail_idx = end_idx
            main_log.info(f"  *** First failure at features[0:{end_idx}] ***")
            # Continue to confirm it's consistent

    # Also test features + global_pool and full model
    main_log.info("\n  Testing features + global_pool ...")
    r_pool = run_slice(FeaturesWithPool(model), inputs, "features+global_pool", "block8.log", threshold)
    results.append(r_pool)
    pool_pcc_str = f"{r_pool['pcc']:.6f}" if r_pool['pcc'] is not None else "N/A"
    main_log.info(f"  {'features+global_pool':30s}  PCC={pool_pcc_str}  "
                  f"{'OK' if r_pool['pcc'] is not None and r_pool['pcc'] >= threshold else 'FAIL'}")

    main_log.info("\n  Testing full model (no dropout)...")
    r_full = run_slice(FullModel(model), inputs, "full_model_no_dropout", "block9.log", threshold)
    results.append(r_full)
    full_pcc_str = f"{r_full['pcc']:.6f}" if r_full['pcc'] is not None else "N/A"
    main_log.info(f"  {'full_model_no_dropout':30s}  PCC={full_pcc_str}  "
                  f"{'OK' if r_full['pcc'] is not None and r_full['pcc'] >= threshold else 'FAIL'}")

    # ── Stage 2: drill into the failing section ───────────────────────────────
    main_log.info("\n" + "=" * 62)
    main_log.info("STAGE 2 — drill into failing section")
    main_log.info("=" * 62)

    if first_fail_idx is None:
        main_log.info("  No failure found in stage 1 — issue may be in global_pool or linear.")
        if r_pool['pcc'] is not None and r_pool['pcc'] < threshold:
            main_log.info("  Failure in global_pool — issue is SelectAdaptivePool2d (adaptive avg pool).")
        elif r_full['pcc'] is not None and r_full['pcc'] < threshold:
            main_log.info("  Failure in last_linear — issue is Linear(1536→1000).")
    else:
        # Find the previous passing end_idx
        passing_end = 0
        for end_idx, name, _ in stage1_slices:
            r = next((x for x in results if x['block'] == name), None)
            if r and r['pcc'] is not None and r['pcc'] >= threshold:
                passing_end = end_idx
            elif r and (r['pcc'] is None or r['pcc'] < threshold):
                break

        main_log.info(f"  Narrowing: last PASS at features[0:{passing_end}], "
                      f"first FAIL at features[0:{first_fail_idx}]")
        main_log.info(f"  Drilling into features[{passing_end}:{first_fail_idx}] one block at a time...")

        for i in range(passing_end, first_fail_idx):
            block = list(model.features.children())[i]
            block_name = f"features[{passing_end}:{i+1}]_{type(block).__name__}"
            log_file = f"stage2_feat{i+1}.log"
            m = FeaturesSlice(model.features, i + 1)
            r = run_slice(m, inputs, block_name, log_file, threshold)
            results.append(r)
            pcc_str = f"{r['pcc']:.6f}" if r['pcc'] is not None else "N/A"
            ok = r['pcc'] is not None and r['pcc'] >= threshold
            main_log.info(f"  features[0:{i+1}] {type(block).__name__:20s}  PCC={pcc_str}  {'OK' if ok else 'FAIL ← FAILING BLOCK'}")
            if not ok:
                main_log.info(f"\n  *** FAILING BLOCK IDENTIFIED: features[{i}] = {type(block).__name__} ***")
                main_log.info(f"  Block structure: {block}")
                break

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = LOG_DIR / "pcc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    main_log.info(f"\nFull results saved to {summary_path}")

    # ── Print final table ─────────────────────────────────────────────────────
    main_log.info("\n" + "=" * 62)
    main_log.info("PCC SUMMARY TABLE")
    main_log.info("=" * 62)
    for r in results:
        pcc_str = f"{r['pcc']:.6f}" if r['pcc'] is not None else "    N/A"
        ok = r['pcc'] is not None and r['pcc'] >= threshold
        main_log.info(f"  {r['block']:40s}  PCC={pcc_str}  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
