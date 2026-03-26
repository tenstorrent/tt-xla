#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
RT-DETR R18vd PCC Debug Script
================================
Systematically isolates the exact operation causing PCC drop by progressively
slicing the RT-DETR R18vd ResNet backbone.

Model forward path (backbone only):
  input (pixel_values: [1, 3, 640, 640])
    → embedder      Conv×3 (3→32→32→64) + MaxPool2d(k=3,s=2,p=1)
    → stage[0]      RTDetrResNetStage  (64→64,  2 layers, no downsampling)
    → stage[1]      RTDetrResNetStage  (64→128, 2 layers, shortcut has AvgPool2d(k=2,s=2))
    → stage[2]      RTDetrResNetStage  (128→256, 2 layers, shortcut has AvgPool2d(k=2,s=2))
    → stage[3]      RTDetrResNetStage  (256→512, 2 layers, shortcut has AvgPool2d(k=2,s=2))
    → encoder (AIFI), decoder ...

Suspected op: AvgPool2d(kernel_size=2, stride=2, padding=0) in shortcut of stage[1..3]
— same class of bug as DenseNet121.

Usage:
    cd /proj_sw/user_dev/ctr-lelanchelian/latest_build/tt-xla
    source venv/activate
    python rt_detr_pcc_debug.py

Logs: rt_detr/debug_summary.log  rt_detr/pcc_summary.json
"""

import copy
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch_xla
    import torch_plugin_tt  # noqa: F401
    TT_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] torch_plugin_tt not importable: {e}")
    TT_AVAILABLE = False

LOG_DIR = PROJECT_ROOT / "rt_detr"
LOG_DIR.mkdir(exist_ok=True)

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


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.detach().float().cpu().flatten()
    b_f = b.detach().float().cpu().flatten()
    if torch.allclose(a_f, b_f, rtol=1e-5, atol=1e-5):
        return 1.0
    va = a_f - a_f.mean()
    vb = b_f - b_f.mean()
    denom = va.norm() * vb.norm()
    return float("nan") if float(denom) == 0.0 else float((va @ vb) / denom)


def tensor_stats(t: torch.Tensor) -> Dict:
    f = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(f.min()), "max": float(f.max()),
        "mean": float(f.mean()), "std": float(f.std()),
        "nan": bool(torch.isnan(f).any()), "inf": bool(torch.isinf(f).any()),
    }


def stats_str(t: torch.Tensor) -> str:
    s = tensor_stats(t)
    return (f"shape={s['shape']}  min={s['min']:.4f}  max={s['max']:.4f}  "
            f"mean={s['mean']:.6f}  std={s['std']:.6f}  nan={s['nan']}  inf={s['inf']}")


def run_tt(model: nn.Module, inputs) -> Optional[Any]:
    if not TT_AVAILABLE:
        return None
    try:
        xla_device = torch_xla.device()
        m = copy.deepcopy(model).eval().to(xla_device)
        if isinstance(inputs, torch.Tensor):
            inp = inputs.to(xla_device)
        else:
            inp = {k: v.to(xla_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        compiled = torch.compile(m, backend="tt", options={})
        with torch.no_grad():
            out = compiled(inp) if isinstance(inp, dict) else compiled(inp)
        torch_xla.sync()

        def to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.cpu()
            if isinstance(x, (list, tuple)):
                return type(x)(to_cpu(i) for i in x)
            return x

        return to_cpu(out)
    except Exception:
        main_log.error(traceback.format_exc())
        return None


def run_slice(model: nn.Module, inputs, name: str, log_file: str, threshold=0.99) -> Dict:
    log = _make_logger(name, log_file)
    log.info("=" * 62)
    log.info(f"Block: {name}")
    log.info("=" * 62)
    model.eval()

    with torch.no_grad():
        cpu_out = model(inputs) if isinstance(inputs, torch.Tensor) else model(**inputs)
    cpu_t = cpu_out if isinstance(cpu_out, torch.Tensor) else cpu_out[0]
    log.info(f"CPU: {stats_str(cpu_t)}")

    tt_raw = run_tt(model, inputs)
    if tt_raw is None:
        log.warning("TT run returned None")
        return {"block": name, "pcc": None}

    tt_t = tt_raw if isinstance(tt_raw, torch.Tensor) else tt_raw[0]
    log.info(f"TT : {stats_str(tt_t)}")

    pcc = compute_pcc(cpu_t, tt_t)
    ok = isinstance(pcc, float) and not (pcc != pcc) and pcc >= threshold
    pcc_str = f"{pcc:.6f}"
    log.info(f"PCC = {pcc_str}  [{'OK' if ok else 'FAIL'}]")
    return {"block": name, "pcc": pcc}


# ── Backbone slice wrappers ───────────────────────────────────────────────────

class BackboneSlice(nn.Module):
    """Run embedder + stages[0:n]."""
    def __init__(self, backbone, num_stages: int):
        super().__init__()
        self.embedder = backbone.model.embedder
        self.stages = nn.ModuleList(list(backbone.model.encoder.stages)[:num_stages])

    def forward(self, pixel_values):
        x = self.embedder(pixel_values)
        for stage in self.stages:
            x = stage(x)
        return x


class SingleStage(nn.Module):
    """Run a single RTDetrResNetStage given pre-computed input."""
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return self.stage(x)


class SingleLayer(nn.Module):
    """Run a single RTDetrResNetBasicLayer."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class ShortcutOnly(nn.Module):
    """Run only the shortcut path (AvgPool2d + Conv) of a BasicLayer."""
    def __init__(self, shortcut):
        super().__init__()
        self.shortcut = shortcut

    def forward(self, x):
        return self.shortcut(x)


class AvgPoolOnly(nn.Module):
    """Run only the AvgPool2d from the shortcut."""
    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def forward(self, x):
        return self.pool(x)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    from datasets import load_dataset

    main_log.info("Loading PekingU/rtdetr_r18vd...")
    model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd").eval()
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")

    dataset = load_dataset("huggingface/cats-image")["test"]
    image = dataset[0]["image"]
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    main_log.info(f"Input: {list(pixel_values.shape)}")

    backbone = model.model.backbone
    stages = list(backbone.model.encoder.stages)
    results = []

    # ── Stage 1: backbone slice sweep ────────────────────────────────────────
    main_log.info("\n" + "=" * 62)
    main_log.info("STAGE 1 — backbone slice sweep")
    main_log.info("=" * 62)

    slices = [
        (0, "embedder_only",          "block1.log"),
        (1, "embedder+stage0",        "block2.log"),
        (2, "embedder+stage0+stage1", "block3.log"),
        (3, "..+stage2",              "block4.log"),
        (4, "..+stage3",              "block5.log"),
    ]

    first_fail_stages = None
    for n_stages, name, log_file in slices:
        m = BackboneSlice(backbone, n_stages)
        r = run_slice(m, pixel_values, name, log_file)
        results.append(r)
        pcc_str = f"{r['pcc']:.6f}" if r['pcc'] is not None else "N/A"
        ok = r['pcc'] is not None and not (r['pcc'] != r['pcc']) and r['pcc'] >= 0.99
        main_log.info(f"  {name:35s}  PCC={pcc_str}  {'OK' if ok else 'FAIL'}")
        if not ok and first_fail_stages is None:
            first_fail_stages = n_stages
            main_log.info(f"  *** First failure at n_stages={n_stages} ***")

    if first_fail_stages is None:
        main_log.info("  No failure in backbone slices — issue is in encoder/decoder.")
        return

    # ── Stage 2: failing stage layer-by-layer ────────────────────────────────
    fail_stage_idx = first_fail_stages - 1  # 0-indexed
    main_log.info(f"\n{'='*62}")
    main_log.info(f"STAGE 2 — drill into stage[{fail_stage_idx}] layers")
    main_log.info("=" * 62)

    # Get input to the failing stage
    prefix = BackboneSlice(backbone, fail_stage_idx)
    with torch.no_grad():
        stage_input = prefix(pixel_values)
    main_log.info(f"  Stage[{fail_stage_idx}] input: {stats_str(stage_input)}")

    fail_stage = stages[fail_stage_idx]
    for li, layer in enumerate(fail_stage.layers):
        m = SingleLayer(layer)
        log_file = f"stage{fail_stage_idx}_layer{li}.log"
        r = run_slice(m, stage_input, f"stage[{fail_stage_idx}].layer[{li}]", log_file)
        results.append(r)
        pcc_str = f"{r['pcc']:.6f}" if r['pcc'] is not None else "N/A"
        ok = r['pcc'] is not None and not (r['pcc'] != r['pcc']) and r['pcc'] >= 0.99
        main_log.info(f"  stage[{fail_stage_idx}].layer[{li}]  PCC={pcc_str}  {'OK' if ok else 'FAIL ← FAILING LAYER'}")
        if not ok:
            # Drill into shortcut vs main path
            main_log.info(f"\n{'='*62}")
            main_log.info(f"STAGE 3 — drill into layer[{li}] shortcut vs main path")
            main_log.info("=" * 62)

            shortcut = getattr(layer, 'shortcut', None)
            if shortcut is not None:
                main_log.info(f"  Shortcut: {shortcut}")
                r_sc = run_slice(ShortcutOnly(shortcut), stage_input,
                                 f"stage[{fail_stage_idx}].layer[{li}].shortcut", f"stage{fail_stage_idx}_layer{li}_shortcut.log")
                results.append(r_sc)
                pcc_str = f"{r_sc['pcc']:.6f}" if r_sc['pcc'] is not None else "N/A"
                ok_sc = r_sc['pcc'] is not None and not (r_sc['pcc'] != r_sc['pcc']) and r_sc['pcc'] >= 0.99
                main_log.info(f"  Shortcut PCC={pcc_str}  {'OK' if ok_sc else 'FAIL ← shortcut is broken'}")

                # Check if shortcut[0] is AvgPool2d
                if isinstance(shortcut, nn.Sequential) and isinstance(shortcut[0], nn.AvgPool2d):
                    pool = shortcut[0]
                    main_log.info(f"\n  Testing AvgPool2d alone: {pool}")
                    r_pool = run_slice(AvgPoolOnly(pool), stage_input,
                                       f"avgpool_stage{fail_stage_idx}", f"avgpool_stage{fail_stage_idx}.log")
                    results.append(r_pool)
                    pcc_str = f"{r_pool['pcc']:.6f}" if r_pool['pcc'] is not None else "N/A"
                    ok_pool = r_pool['pcc'] is not None and not (r_pool['pcc'] != r_pool['pcc']) and r_pool['pcc'] >= 0.99
                    main_log.info(f"  AvgPool2d alone PCC={pcc_str}  {'OK' if ok_pool else 'FAIL ← ROOT CAUSE'}")
                    if not ok_pool:
                        main_log.info(f"\n  *** CONFIRMED ROOT CAUSE ***")
                        main_log.info(f"  Op : {pool}")
                        main_log.info(f"  Location: model.backbone.model.encoder.stages[{fail_stage_idx}]")
                        main_log.info(f"            .layers[{li}].shortcut[0]")
            break

    # Save summary
    summary_path = LOG_DIR / "pcc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    main_log.info(f"\nResults saved to {summary_path}")
    main_log.info("\n" + "=" * 62)
    main_log.info("SUMMARY TABLE")
    main_log.info("=" * 62)
    for r in results:
        pcc_str = f"{r['pcc']:.6f}" if r['pcc'] is not None else "    N/A"
        ok = r['pcc'] is not None and not (r['pcc'] != r['pcc']) and r['pcc'] >= 0.99
        main_log.info(f"  {r['block']:45s}  PCC={pcc_str}  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
