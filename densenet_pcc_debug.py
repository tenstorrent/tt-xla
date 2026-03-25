#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
DenseNet121 PCC Debug Script
=============================
Systematically isolates the exact operation causing PCC drop by progressively
slicing the DenseNet121 (torchvision) model.

Model forward path:
  input
    → features.conv0     Conv2d(3→64, 7×7, s=2)
    → features.norm0     BatchNorm2d(64)
    → features.relu0     ReLU
    → features.pool0     MaxPool2d(3×3, s=2)
    → features.denseblock1  _DenseBlock × 6 dense layers  (64→256 ch)
    → features.transition1  BN+ReLU+Conv1×1+AvgPool       (256→128 ch)
    → features.denseblock2  _DenseBlock × 12 dense layers (128→512 ch)
    → features.transition2  BN+ReLU+Conv1×1+AvgPool       (512→256 ch)
    → features.denseblock3  _DenseBlock × 24 dense layers (256→1024 ch)
    → features.transition3  BN+ReLU+Conv1×1+AvgPool       (1024→512 ch)
    → features.denseblock4  _DenseBlock × 16 dense layers (512→1024 ch)
    → features.norm5     BatchNorm2d(1024)
    → F.relu + F.adaptive_avg_pool2d
    → classifier         Linear(1024→1000)

Usage:
    cd /proj_sw/user_dev/ctr-lelanchelian/latest_build/tt-xla
    source venv/activate
    python densenet_pcc_debug.py

Logs:  densenet/block1.log  block2.log  ...  layer1.log  layer2.log  ...
       densenet/debug_summary.log
       densenet/pcc_summary.json
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
LOG_DIR = PROJECT_ROOT / "densenet"
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


def _to_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    tensors: List[torch.Tensor] = []

    def _collect(x):
        if isinstance(x, torch.Tensor):
            tensors.append(x.detach().float().cpu().flatten())
        elif isinstance(x, (list, tuple)):
            for item in x:
                _collect(item)
        elif isinstance(x, dict):
            for v in x.values():
                _collect(v)

    _collect(out)
    if not tensors:
        raise ValueError(f"No tensors found in output of type {type(out)}")
    return torch.cat(tensors)


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
        cpu_tensor = _to_tensor(cpu_out)
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
        tt_tensor = _to_tensor(tt_out_raw)
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
# Block-level slice wrappers
# ══════════════════════════════════════════════════════════════════════════════


class _SliceStem(nn.Module):
    """conv0 only (Conv7×7)"""
    def __init__(self, model):
        super().__init__()
        self.conv0 = model.features.conv0

    def forward(self, x):
        return self.conv0(x)


class _SliceStemFull(nn.Module):
    """conv0 + norm0 + relu0 + pool0"""
    def __init__(self, model):
        super().__init__()
        self.conv0 = model.features.conv0
        self.norm0 = model.features.norm0
        self.relu0 = model.features.relu0
        self.pool0 = model.features.pool0

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        return x


class _SliceUpTo(nn.Module):
    """features up to and including named_block (sequential prefix)"""

    # Ordered list of feature block names in DenseNet121
    _BLOCKS = [
        "conv0", "norm0", "relu0", "pool0",
        "denseblock1", "transition1",
        "denseblock2", "transition2",
        "denseblock3", "transition3",
        "denseblock4", "norm5",
    ]

    def __init__(self, model, up_to: str, include_relu_pool: bool = False,
                 include_classifier: bool = False):
        """
        include_relu_pool: apply F.relu + F.adaptive_avg_pool2d + flatten after norm5
        include_classifier: also apply classifier Linear
        """
        super().__init__()
        assert up_to in self._BLOCKS, f"Unknown block: {up_to}"
        idx = self._BLOCKS.index(up_to)
        selected = self._BLOCKS[: idx + 1]
        for name in selected:
            setattr(self, name, getattr(model.features, name))
        self._selected = selected
        self._include_relu_pool = include_relu_pool
        self._include_classifier = include_classifier
        if include_classifier:
            self.classifier = model.classifier

    def forward(self, x):
        for name in self._selected:
            x = getattr(self, name)(x)
        if self._include_relu_pool:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        if self._include_classifier:
            x = self.classifier(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Dense-layer-level slicer — drills into a _DenseBlock layer by layer
# ══════════════════════════════════════════════════════════════════════════════


class _SliceDenseBlockPartial(nn.Module):
    """
    Runs features up to (but not including) the target denseblock, then runs
    that denseblock's first N dense layers only.
    """

    def __init__(self, model, block_name: str, n_layers: int):
        super().__init__()
        assert block_name in ("denseblock1", "denseblock2", "denseblock3", "denseblock4")
        # Collect all feature modules before this block
        all_blocks = ["conv0", "norm0", "relu0", "pool0",
                      "denseblock1", "transition1",
                      "denseblock2", "transition2",
                      "denseblock3", "transition3",
                      "denseblock4", "norm5"]
        prefix = all_blocks[: all_blocks.index(block_name)]
        for nm in prefix:
            setattr(self, nm, getattr(model.features, nm))
        self._prefix = prefix
        self.block_name = block_name

        # Grab only the first n_layers dense layers from the target block
        target_block = getattr(model.features, block_name)
        self._dense_layers = nn.ModuleList()
        for i, (_, layer) in enumerate(target_block.named_children()):
            if i >= n_layers:
                break
            self._dense_layers.append(layer)
        self._n_layers = n_layers

    def forward(self, x):
        # Run the prefix blocks
        for nm in self._prefix:
            x = getattr(self, nm)(x)

        # Run first n_layers dense layers manually (mimicking _DenseBlock.forward)
        features = [x]
        for layer in self._dense_layers:
            new_feat = layer(features)
            features.append(new_feat)
        return torch.cat(features, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Sub-op slicer inside a single _DenseLayer
# ══════════════════════════════════════════════════════════════════════════════


class _SliceDenseLayerSubOp(nn.Module):
    """
    Runs the full network up to (and including) a specific sub-op inside
    dense layer N of target denseblock.

    sub_op choices: "norm1", "relu1", "conv1", "norm2", "relu2", "conv2"
    (these are the sequential ops inside each _DenseLayer)
    """

    _SUB_OPS = ("norm1", "relu1", "conv1", "norm2", "relu2", "conv2")

    def __init__(self, model, block_name: str, layer_idx: int, sub_op: str):
        super().__init__()
        assert sub_op in self._SUB_OPS, f"Unknown sub_op: {sub_op}"
        all_blocks = ["conv0", "norm0", "relu0", "pool0",
                      "denseblock1", "transition1",
                      "denseblock2", "transition2",
                      "denseblock3", "transition3",
                      "denseblock4", "norm5"]
        prefix = all_blocks[: all_blocks.index(block_name)]
        for nm in prefix:
            setattr(self, nm, getattr(model.features, nm))
        self._prefix = prefix

        target_block = getattr(model.features, block_name)
        layers = list(target_block.named_children())
        # layers before target_layer_idx run in full
        self._prior_layers = nn.ModuleList([l for _, l in layers[:layer_idx]])
        # the target layer
        self._target_layer = layers[layer_idx][1]
        self._sub_op = sub_op

    def forward(self, x):
        for nm in self._prefix:
            x = getattr(self, nm)(x)

        # Run prior dense layers
        features = [x]
        for layer in self._prior_layers:
            new_feat = layer(features)
            features.append(new_feat)

        # Now probe sub-ops inside target dense layer
        concat = torch.cat(features, 1)
        tl = self._target_layer

        out = tl.norm1(concat)
        if self._sub_op == "norm1":
            return out
        out = tl.relu1(out)
        if self._sub_op == "relu1":
            return out
        out = tl.conv1(out)
        if self._sub_op == "conv1":
            return out
        out = tl.norm2(out)
        if self._sub_op == "norm2":
            return out
        out = tl.relu2(out)
        if self._sub_op == "relu2":
            return out
        out = tl.conv2(out)
        return out   # sub_op == "conv2"


# ══════════════════════════════════════════════════════════════════════════════
# Transition sub-op slicer
# ══════════════════════════════════════════════════════════════════════════════


class _SliceTransitionSubOp(nn.Module):
    """
    Runs features up to (but not including) target_transition, then probes
    sub-ops within that transition block.

    sub_op choices: "norm", "relu", "conv", "pool" (the four ops in _Transition)
    """

    _SUB_OPS = ("norm", "relu", "conv", "pool")

    def __init__(self, model, transition_name: str, sub_op: str):
        super().__init__()
        assert sub_op in self._SUB_OPS, f"Unknown sub_op: {sub_op}"
        all_blocks = ["conv0", "norm0", "relu0", "pool0",
                      "denseblock1", "transition1",
                      "denseblock2", "transition2",
                      "denseblock3", "transition3",
                      "denseblock4", "norm5"]
        prefix = all_blocks[: all_blocks.index(transition_name)]
        for nm in prefix:
            setattr(self, nm, getattr(model.features, nm))
        self._prefix = prefix
        # Store transition sub-modules
        t = getattr(model.features, transition_name)
        self.t_norm = t.norm
        self.t_relu = t.relu
        self.t_conv = t.conv
        self.t_pool = t.pool
        self._sub_op = sub_op

    def forward(self, x):
        for nm in self._prefix:
            x = getattr(self, nm)(x)
        x = self.t_norm(x)
        if self._sub_op == "norm":
            return x
        x = self.t_relu(x)
        if self._sub_op == "relu":
            return x
        x = self.t_conv(x)
        if self._sub_op == "conv":
            return x
        x = self.t_pool(x)
        return x   # sub_op == "pool"


# ══════════════════════════════════════════════════════════════════════════════
# Analysis passes
# ══════════════════════════════════════════════════════════════════════════════

THRESHOLD = 0.99


def _print_row(r: Dict) -> None:
    pcc = r["pcc"]
    th = r.get("threshold", THRESHOLD)
    pcc_s = f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "N/A"
    ok = isinstance(pcc, float) and pcc == pcc and pcc >= th
    flag = "OK  " if ok else "FAIL"
    main_log.info(f"  [{flag}]  {r['block']:<56}  PCC={pcc_s}  ({r['log_file']})")


def _first_fail(results: List[Dict]) -> Optional[str]:
    for r in results:
        pcc = r["pcc"]
        th = r.get("threshold", THRESHOLD)
        if isinstance(pcc, float) and (pcc != pcc or pcc < th):
            return r["block"]
    return None


def block_level_analysis(model, inputs) -> List[Dict]:
    main_log.info("\n" + "=" * 62)
    main_log.info("PHASE 1 — Block-level PCC analysis")
    main_log.info("=" * 62)
    results = []
    bidx = 1

    def _run(slicer, name):
        nonlocal bidx
        r = run_slice(slicer, inputs, name, f"block{bidx}.log", threshold=THRESHOLD)
        results.append(r)
        bidx += 1
        _print_row(r)

    # ── Stem: conv0 only ──────────────────────────────────────────────────────
    _run(_SliceStem(model), "stem.conv0")

    # ── Full stem (conv0+norm0+relu0+pool0) ────────────────────────────────────
    _run(_SliceStemFull(model), "stem (conv0→pool0)")

    # ── Progressive feature blocks ────────────────────────────────────────────
    for block_name in ["denseblock1", "transition1",
                       "denseblock2", "transition2",
                       "denseblock3", "transition3",
                       "denseblock4", "norm5"]:
        _run(_SliceUpTo(model, block_name), f"features up to {block_name}")

    # ── With global pool (features + relu + avgpool + flatten) ───────────────
    _run(_SliceUpTo(model, "norm5", include_relu_pool=True), "features + relu + avgpool")

    # ── Full model including classifier ────────────────────────────────────────
    _run(_SliceUpTo(model, "norm5", include_relu_pool=True, include_classifier=True),
         "full model (with classifier)")

    return results


def transition_sub_op_analysis(model, inputs, transition_name: str) -> List[Dict]:
    """Drills into a _Transition block sub-op by sub-op."""
    main_log.info("\n" + "=" * 62)
    main_log.info(f"PHASE 2b — Transition sub-op PCC in {transition_name}")
    main_log.info("=" * 62)

    results = []
    tidx = 1

    for sub_op in _SliceTransitionSubOp._SUB_OPS:
        name = f"{transition_name}.{sub_op}"
        try:
            slicer = _SliceTransitionSubOp(model, transition_name, sub_op)
            r = run_slice(slicer, inputs, name, f"trans{tidx}.log", threshold=THRESHOLD)
        except Exception:
            main_log.error(f"  ERROR building slicer for {name}:\n{traceback.format_exc()}")
            r = {"block": name, "pcc": None, "log_file": f"trans{tidx}.log",
                 "cpu_stats": None, "tt_stats": None, "threshold": THRESHOLD}
        results.append(r)
        tidx += 1
        _print_row(r)

    return results


def dense_layer_level_analysis(model, inputs, block_name: str) -> List[Dict]:
    """Drills into target denseblock layer by layer."""
    target_block = getattr(model.features, block_name)
    n_total = sum(1 for _ in target_block.named_children())

    main_log.info("\n" + "=" * 62)
    main_log.info(f"PHASE 2 — Dense-layer-level PCC analysis in {block_name}")
    main_log.info(f"          ({n_total} dense layers)")
    main_log.info("=" * 62)

    results = []
    lidx = 1

    def _run(slicer, name):
        nonlocal lidx
        r = run_slice(slicer, inputs, name, f"layer{lidx}.log", threshold=THRESHOLD)
        results.append(r)
        lidx += 1
        _print_row(r)

    for n in range(1, n_total + 1):
        _run(
            _SliceDenseBlockPartial(model, block_name, n),
            f"{block_name} first {n} layer(s)",
        )

    return results


def sub_op_analysis(model, inputs, block_name: str, layer_idx: int) -> List[Dict]:
    """Sub-op PCC inside one _DenseLayer."""
    main_log.info("\n" + "=" * 62)
    main_log.info(f"PHASE 3 — Sub-op PCC in {block_name}.denselayer{layer_idx+1}")
    main_log.info("=" * 62)

    results = []
    oidx = 1

    def _run(slicer, name):
        nonlocal oidx
        r = run_slice(slicer, inputs, name, f"op{oidx}.log", threshold=THRESHOLD)
        results.append(r)
        oidx += 1
        _print_row(r)

    for sub_op in _SliceDenseLayerSubOp._SUB_OPS:
        name = f"{block_name}.denselayer{layer_idx+1}.{sub_op}"
        try:
            slicer = _SliceDenseLayerSubOp(model, block_name, layer_idx, sub_op)
            r = run_slice(slicer, inputs, name, f"op{oidx}.log", threshold=THRESHOLD)
        except Exception:
            main_log.error(f"  ERROR building slicer for {name}:\n{traceback.format_exc()}")
            r = {"block": name, "pcc": None, "log_file": f"op{oidx}.log",
                 "cpu_stats": None, "tt_stats": None, "threshold": THRESHOLD}
        results.append(r)
        oidx += 1
        _print_row(r)

    return results


def print_final_report(
    block_results: List[Dict],
    layer_results: Optional[List[Dict]],
    op_results: Optional[List[Dict]],
    trans_results: Optional[List[Dict]] = None,
) -> None:
    sep = "=" * 72

    main_log.info("\n" + sep)
    main_log.info("BLOCK-LEVEL PCC TABLE")
    main_log.info(sep)
    main_log.info(f"  {'Status':<6}  {'Block':<58}  {'PCC':>10}  Log")
    main_log.info("  " + "-" * 70)
    for r in block_results:
        pcc = r["pcc"]
        th = r.get("threshold", THRESHOLD)
        pcc_s = f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "N/A"
        ok = isinstance(pcc, float) and pcc == pcc and pcc >= th
        status = "OK  " if ok else "FAIL"
        main_log.info(f"  [{status}]  {r['block']:<58}  {pcc_s:>10}  {r['log_file']}")

    if layer_results:
        main_log.info("\n" + sep)
        main_log.info("DENSE-LAYER-LEVEL PCC TABLE")
        main_log.info(sep)
        main_log.info(f"  {'Status':<6}  {'Layer':<58}  {'PCC':>10}  Log")
        main_log.info("  " + "-" * 70)
        for r in layer_results:
            pcc = r["pcc"]
            th = r.get("threshold", THRESHOLD)
            pcc_s = f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "N/A"
            ok = isinstance(pcc, float) and pcc == pcc and pcc >= th
            status = "OK  " if ok else "FAIL"
            main_log.info(f"  [{status}]  {r['block']:<58}  {pcc_s:>10}  {r['log_file']}")

    if op_results:
        main_log.info("\n" + sep)
        main_log.info("SUB-OP PCC TABLE (failing dense layer)")
        main_log.info(sep)
        main_log.info(f"  {'Status':<6}  {'Op':<58}  {'PCC':>10}  Log")
        main_log.info("  " + "-" * 70)
        for r in op_results:
            pcc = r["pcc"]
            th = r.get("threshold", THRESHOLD)
            pcc_s = f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "N/A"
            ok = isinstance(pcc, float) and pcc == pcc and pcc >= th
            status = "OK  " if ok else "FAIL"
            cpu_s = r.get("cpu_stats") or {}
            tt_s = r.get("tt_stats") or {}
            extra = ""
            if cpu_s and tt_s:
                extra = (
                    f"  CPU std={cpu_s.get('std', 0):.4f} max={cpu_s.get('max', 0):.4f}"
                    f"  TT std={tt_s.get('std', 0):.4f} max={tt_s.get('max', 0):.4f}"
                )
            main_log.info(f"  [{status}]  {r['block']:<58}  {pcc_s:>10}  {r['log_file']}{extra}")

    if trans_results:
        main_log.info("\n" + sep)
        main_log.info("TRANSITION SUB-OP PCC TABLE")
        main_log.info(sep)
        main_log.info(f"  {'Status':<6}  {'Op':<58}  {'PCC':>10}  Log")
        main_log.info("  " + "-" * 70)
        for r in trans_results:
            pcc = r["pcc"]
            th = r.get("threshold", THRESHOLD)
            pcc_s = f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "N/A"
            ok = isinstance(pcc, float) and pcc == pcc and pcc >= th
            status = "OK  " if ok else "FAIL"
            cpu_s = r.get("cpu_stats") or {}
            tt_s = r.get("tt_stats") or {}
            extra = ""
            if cpu_s and tt_s:
                extra = (
                    f"  CPU std={cpu_s.get('std', 0):.4f} max={cpu_s.get('max', 0):.4f}"
                    f"  TT std={tt_s.get('std', 0):.4f} max={tt_s.get('max', 0):.4f}"
                    f"  TT nan={tt_s.get('has_nan', '?')} inf={tt_s.get('has_inf', '?')}"
                )
            main_log.info(f"  [{status}]  {r['block']:<58}  {pcc_s:>10}  {r['log_file']}{extra}")

    failing_block = _first_fail(block_results)
    failing_layer = _first_fail(layer_results) if layer_results else None
    failing_op = _first_fail(op_results) if op_results else None
    failing_trans = _first_fail(trans_results) if trans_results else None

    main_log.info("\n" + sep)
    main_log.info(f"  FIRST FAILING BLOCK       : {failing_block or '— all blocks passed —'}")
    main_log.info(f"  FIRST FAILING TRANSITION  : {failing_trans or '— all transition ops passed —'}")
    main_log.info(f"  FIRST FAILING LAYER       : {failing_layer or '— all layers passed —'}")
    main_log.info(f"  FIRST FAILING SUB-OP      : {failing_op    or '— all sub-ops passed —'}")
    main_log.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def main():
    main_log.info("=" * 72)
    main_log.info("DenseNet121 PCC Debug Script")
    main_log.info(f"Log directory : {LOG_DIR}")
    main_log.info(f"TT available  : {TT_AVAILABLE}")
    main_log.info("=" * 72)

    # ── load model + inputs ───────────────────────────────────────────────────
    main_log.info("Loading DenseNet121 model …")
    from third_party.tt_forge_models.densenet.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    loader = ModelLoader(variant=ModelVariant.DENSENET121)
    model = loader.load_model()
    inputs = loader.load_inputs()
    model.eval()
    main_log.info(f"Model loaded.  Input shape: {inputs.shape}  dtype: {inputs.dtype}")
    main_log.info(
        f"Model params : {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M"
    )

    # ── Phase 1: block-level ──────────────────────────────────────────────────
    block_results = block_level_analysis(model, inputs)
    failing_block = _first_fail(block_results)
    main_log.info(f"\nFirst failing block → {failing_block or 'None (all OK)'}")

    layer_results = None
    op_results = None
    trans_results = None

    transition_names = ["transition1", "transition2", "transition3"]
    denseblock_names = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]

    target_transition = None
    target_block = None

    if failing_block:
        # Check if the failing block is a transition
        for tname in transition_names:
            if tname in failing_block:
                target_transition = tname
                break

        if target_transition:
            # Phase 2b: drill into transition sub-ops
            main_log.info(f"\nDrilling into {target_transition} sub-ops …")
            trans_results = transition_sub_op_analysis(model, inputs, target_transition)
        else:
            # Failing at a denseblock itself — drill into layers
            for bname in denseblock_names:
                if bname in failing_block:
                    target_block = bname
                    break
            # If failing at norm5/relu/avgpool/classifier — preceding denseblock
            if target_block is None:
                fail_idx = next((i for i, r in enumerate(block_results)
                                 if r["block"] == failing_block), None)
                if fail_idx is not None:
                    for bname in reversed(denseblock_names):
                        ref = f"features up to {bname}"
                        ref_idx = next((i for i, r in enumerate(block_results)
                                        if r["block"] == ref), None)
                        if ref_idx is not None and ref_idx <= fail_idx:
                            target_block = bname
                            break

    if target_block:
        main_log.info(f"\nDrilling into {target_block} …")
        layer_results = dense_layer_level_analysis(model, inputs, target_block)
        failing_layer = _first_fail(layer_results)
        main_log.info(f"\nFirst failing dense layer → {failing_layer or 'None'}")

        if failing_layer:
            fail_n = next(
                (i for i, r in enumerate(layer_results) if r["block"] == failing_layer),
                None,
            )
            if fail_n is not None:
                main_log.info(f"\nDrilling into sub-ops of {target_block}.denselayer{fail_n+1} …")
                op_results = sub_op_analysis(model, inputs, target_block, fail_n)
    elif target_transition is None and not target_block:
        main_log.info("\nNo denseblock/transition identified — check stem or classifier layers.")

    # ── Final report ──────────────────────────────────────────────────────────
    print_final_report(block_results, layer_results, op_results, trans_results)

    # ── Save JSON summary ─────────────────────────────────────────────────────
    def _jsonify(results):
        if not results:
            return []
        out = []
        for r in results:
            entry = {k: v for k, v in r.items() if k not in ("cpu_stats", "tt_stats")}
            entry["cpu_stats"] = r.get("cpu_stats")
            entry["tt_stats"] = r.get("tt_stats")
            out.append(entry)
        return out

    summary = {
        "block_results": _jsonify(block_results),
        "trans_results": _jsonify(trans_results or []),
        "layer_results": _jsonify(layer_results or []),
        "op_results": _jsonify(op_results or []),
        "first_failing_block": _first_fail(block_results),
        "first_failing_transition": _first_fail(trans_results) if trans_results else None,
        "first_failing_layer": _first_fail(layer_results) if layer_results else None,
        "first_failing_op": _first_fail(op_results) if op_results else None,
    }
    summary_path = LOG_DIR / "pcc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    main_log.info(f"\nJSON summary written to {summary_path}")


if __name__ == "__main__":
    main()
