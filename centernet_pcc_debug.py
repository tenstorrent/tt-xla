#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
CenterNet DLA-1x PCC Debug Script
==================================
Systematically isolates the exact operation causing PCC drop by progressively
slicing the DLASeg (CenterNet DLA-34) model.

Model forward path:
  input
    → base (DLA-34): base_layer → level0 → level1 → level2 → level3 → level4 → level5
    → dla_up (DLAUp): ida_0 → ida_1 → ida_2   (each modifies feature pyramid in-place)
    → ida_up (IDAUp): step-1 → step-2          (further upsampling)
    → heads: hm  /  wh  /  reg

Usage:
    cd /proj_sw/user_dev/ctr-lelanchelian/latest_build/tt-xla
    source venv/activate
    python centernet_pcc_debug.py

Logs:  centernet/block1.log  block2.log  ...  op1.log  op2.log  ...
       centernet/debug_summary.log
       centernet/pcc_summary.json
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

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── register TT backend early ─────────────────────────────────────────────────
# torch_xla must be imported BEFORE torch_plugin_tt to avoid a circular import:
# torch_plugin_tt → torch_xla → plugins.register_installed_plugins() → torch_plugin_tt.TTPlugin
# (which doesn't exist yet because torch_plugin_tt is still initialising).
try:
    import torch_xla              # 1st: triggers plugin discovery via entry_points
    import torch_plugin_tt  # noqa: F401  # 2nd: already in sys.modules, just re-exports

    TT_AVAILABLE = True
except ImportError as _e:
    print(f"[WARNING] torch_plugin_tt not importable: {_e}  — TT runs will be skipped.")
    TT_AVAILABLE = False

# ── output directory ──────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "centernet"
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
    """Flatten any nested output into a single 1-D tensor for PCC."""
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
        # Deep-copy to avoid mutating shared submodules, then move to XLA device.
        # The actual test runner does: workload.model = workload.model.to(device)
        # before compilation, otherwise BN buffers (running_mean/var) stay on CPU
        # while the traced graph runs on xla:0, causing a device mismatch during
        # torch.export.export's FakeTensor propagation.
        xla_device = torch_xla.device()
        model_xla = copy.deepcopy(model)
        model_xla.eval()
        model_xla = model_xla.to(xla_device)
        inputs_xla = inputs.to(xla_device)

        model_tt = torch.compile(model_xla, backend="tt", options={})
        with torch.no_grad():
            out = model_tt(inputs_xla)
        torch_xla.sync()

        # Move result back to CPU
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


# ── block / op result helper ──────────────────────────────────────────────────


def run_slice(
    slice_model: nn.Module,
    inputs: torch.Tensor,
    block_name: str,
    log_file: str,
) -> Dict:
    log = _make_logger(block_name, log_file)
    log.info("=" * 62)
    log.info(f"Block : {block_name}")
    log.info("=" * 62)

    slice_model.eval()

    # --- CPU reference ---
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

    # --- TT run ---
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
    ok = isinstance(pcc, float) and not (pcc != pcc) and pcc >= 0.97  # not nan, >= 0.97
    status = "OK  " if ok else "FAIL"
    log.info(f"PCC = {pcc:.6f}  [{status}]")
    return {
        "block": block_name,
        "pcc": pcc,
        "log_file": log_file,
        "cpu_stats": cpu_stats,
        "tt_stats": tt_stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Block-level slice wrappers
# Each takes the raw input tensor and returns ONE tensor for PCC comparison.
# ══════════════════════════════════════════════════════════════════════════════


class _SliceBaseLayer(nn.Module):
    """base_layer only  (Conv7×7 → BN → ReLU)"""

    def __init__(self, model):
        super().__init__()
        self.base_layer = model.base.base_layer

    def forward(self, x):
        return self.base_layer(x)


class _SliceBaseUpToLevel(nn.Module):
    """base_layer + level0 .. levelN  (N = 0..5)"""

    def __init__(self, model, n: int):
        super().__init__()
        self.base_layer = model.base.base_layer
        self.levels = nn.ModuleList(
            [getattr(model.base, f"level{i}") for i in range(n + 1)]
        )

    def forward(self, x):
        x = self.base_layer(x)
        for lvl in self.levels:
            x = lvl(x)
        return x


class _SliceDLAUpPartial(nn.Module):
    """Full DLA base + DLAUp up to n_ida IDA iterations  (n_ida = 1..3)"""

    def __init__(self, model, n_ida: int):
        super().__init__()
        self.base = model.base
        self.dla_up = model.dla_up
        self.n_ida = n_ida

    def forward(self, x):
        layers = []
        x = self.base.base_layer(x)
        for i in range(6):
            x = getattr(self.base, f"level{i}")(x)
            layers.append(x)
        # Partial DLAUp: run only n_ida IDA blocks
        dla = self.dla_up
        max_iter = len(layers) - dla.startp - 1  # 6 - 2 - 1 = 3
        for i in range(min(self.n_ida, max_iter)):
            ida = getattr(dla, f"ida_{i}")
            ida(layers, len(layers) - i - 2, len(layers))
        return layers[-1]


class _SliceIDAUpPartial(nn.Module):
    """Full DLA base + full DLAUp + IDAUp partial (n_steps iterations)"""

    def __init__(self, model, n_steps: int):
        super().__init__()
        self.base = model.base
        self.dla_up = model.dla_up
        self.ida_up = model.ida_up
        self.first_level = model.first_level
        self.last_level = model.last_level
        self.n_steps = n_steps

    def forward(self, x):
        layers = []
        x = self.base.base_layer(x)
        for i in range(6):
            x = getattr(self.base, f"level{i}")(x)
            layers.append(x)
        x_up = self.dla_up(layers)
        # build y list for ida_up
        n = self.last_level - self.first_level   # = 5 - 2 = 3
        y = [x_up[i].clone() for i in range(n)]
        # partial ida_up
        ida = self.ida_up
        steps_done = 0
        for i in range(1, n):
            if steps_done >= self.n_steps:
                break
            y[i] = getattr(ida, f"up_{i}")(getattr(ida, f"proj_{i}")(y[i]))
            y[i] = getattr(ida, f"node_{i}")(y[i] + y[i - 1])
            steps_done += 1
        return y[-1]


class _SliceWithHead(nn.Module):
    """Full backbone (base + DLAUp + IDAUp) + one detection head."""

    def __init__(self, model, head_name: str):
        super().__init__()
        self.base = model.base
        self.dla_up = model.dla_up
        self.ida_up = model.ida_up
        self.head = getattr(model, head_name)
        self.first_level = model.first_level
        self.last_level = model.last_level

    def forward(self, x):
        layers = []
        x = self.base.base_layer(x)
        for i in range(6):
            x = getattr(self.base, f"level{i}")(x)
            layers.append(x)
        x_up = self.dla_up(layers)
        n = self.last_level - self.first_level
        y = [x_up[i].clone() for i in range(n)]
        self.ida_up(y, 0, len(y))
        return self.head(y[-1])


# ══════════════════════════════════════════════════════════════════════════════
# Fine-grained op-level slicers  (IDAUp sub-ops)
# ══════════════════════════════════════════════════════════════════════════════
#
# Within each IDAUp step i (1-indexed):
#   op_A: proj_i.conv (DCN.forward) → raw deformable conv output
#   op_B: proj_i.actf            → BN + ReLU on proj output  (= full DeformConv)
#   op_C: up_i (ConvTranspose2d) → upsampled feature
#   op_D: add (y[i] + y[i-1])   → residual sum before node
#   op_E: node_i.conv (DCN)      → raw deformable conv output
#   op_F: node_i.actf            → BN + ReLU on node output  (= full node DeformConv = y[i])
#


class _SliceIDAUpSubOp(nn.Module):
    """
    Runs the full backbone up to a sub-operation inside a specific IDAUp step.

    Parameters
    ----------
    model      : full DLASeg model
    target_step: which IDA step to probe (1-indexed, 1 or 2 for a typical 3-element y)
    sub_op     : one of  "proj_dcn", "proj_actf", "up", "add", "node_dcn", "node_actf"
    """

    _SUB_OPS = ("proj_dcn", "proj_actf", "up", "add", "node_dcn", "node_actf")

    def __init__(self, model, target_step: int, sub_op: str):
        super().__init__()
        assert sub_op in self._SUB_OPS, f"Unknown sub_op: {sub_op}"
        self.base = model.base
        self.dla_up = model.dla_up
        self.ida_up = model.ida_up
        self.first_level = model.first_level
        self.last_level = model.last_level
        self.target_step = target_step
        self.sub_op = sub_op

    @staticmethod
    def _dcn_forward(dcn_module, x):
        """Run only the DCN kernel (skip actf)."""
        import torchvision
        om = dcn_module.conv_offset_mask(x)
        o1, o2, mask_raw = torch.chunk(om, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask_raw)
        return torchvision.ops.deform_conv2d(
            x,
            offset,
            dcn_module.weight,
            dcn_module.bias,
            stride=dcn_module.stride,
            padding=dcn_module.padding,
            dilation=dcn_module.dilation,
            mask=mask,
        )

    def forward(self, x):
        layers = []
        x = self.base.base_layer(x)
        for i in range(6):
            x = getattr(self.base, f"level{i}")(x)
            layers.append(x)
        x_up = self.dla_up(layers)
        n = self.last_level - self.first_level
        y = [x_up[i].clone() for i in range(n)]

        ida = self.ida_up
        for step in range(1, n):
            if step < self.target_step:
                # Run full step
                y[step] = getattr(ida, f"up_{step}")(getattr(ida, f"proj_{step}")(y[step]))
                y[step] = getattr(ida, f"node_{step}")(y[step] + y[step - 1])
                continue

            # ---- target step: probe sub-ops one by one ----
            proj_mod = getattr(ida, f"proj_{step}")  # DeformConv
            up_mod = getattr(ida, f"up_{step}")       # ConvTranspose2d
            node_mod = getattr(ida, f"node_{step}")   # DeformConv

            if self.sub_op == "proj_dcn":
                return self._dcn_forward(proj_mod.conv, y[step])

            proj_out = proj_mod(y[step])   # full DeformConv (dcn + actf)
            if self.sub_op == "proj_actf":
                return proj_out

            up_out = up_mod(proj_out)
            if self.sub_op == "up":
                return up_out

            add_out = up_out + y[step - 1]
            if self.sub_op == "add":
                return add_out

            if self.sub_op == "node_dcn":
                return self._dcn_forward(node_mod.conv, add_out)

            # node_actf = full node DeformConv
            return node_mod(add_out)

        return y[-1]


# ══════════════════════════════════════════════════════════════════════════════
# Fine-grained slicer for detection head sub-ops
# ══════════════════════════════════════════════════════════════════════════════
#
# Each detection head (hm / wh / reg) is:
#   nn.Sequential(
#       [0] Conv2d(64, head_conv, 3, padding=1, bias=True),   ← "conv1"
#       [1] ReLU(inplace=True),                                ← "relu"
#       [2] Conv2d(head_conv, classes, 1, bias=True),          ← "conv2"
#   )
#
# sub_op choices: "conv1", "relu", "conv2"


class _SliceHeadSubOp(nn.Module):
    """
    Full backbone + sub-op slice inside a detection head.

    Parameters
    ----------
    model    : full DLASeg model
    head_name: "hm", "wh", or "reg"
    sub_op   : "conv1" | "relu" | "conv2"
    """

    _SUB_OPS = ("conv1", "relu", "conv2")

    def __init__(self, model, head_name: str, sub_op: str):
        super().__init__()
        assert sub_op in self._SUB_OPS, f"Unknown sub_op: {sub_op}"
        self.base = model.base
        self.dla_up = model.dla_up
        self.ida_up = model.ida_up
        self.head = getattr(model, head_name)   # nn.Sequential
        self.first_level = model.first_level
        self.last_level = model.last_level
        self.sub_op = sub_op

    def _backbone_feat(self, x):
        layers = []
        x = self.base.base_layer(x)
        for i in range(6):
            x = getattr(self.base, f"level{i}")(x)
            layers.append(x)
        x_up = self.dla_up(layers)
        n = self.last_level - self.first_level
        y = [x_up[i].clone() for i in range(n)]
        self.ida_up(y, 0, len(y))
        return y[-1]

    def forward(self, x):
        feat = self._backbone_feat(x)
        # head[0] = Conv2d  head[1] = ReLU  head[2] = Conv2d
        out = self.head[0](feat)          # conv1
        if self.sub_op == "conv1":
            return out
        out = self.head[1](out)           # relu
        if self.sub_op == "relu":
            return out
        return self.head[2](out)          # conv2  (= full head output)


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis passes
# ══════════════════════════════════════════════════════════════════════════════


def block_level_analysis(model, inputs) -> List[Dict]:
    main_log.info("\n" + "=" * 62)
    main_log.info("PHASE 1 — Block-level PCC analysis")
    main_log.info("=" * 62)
    results = []
    bidx = 1  # log-file counter

    def _run(slicer, name):
        nonlocal bidx
        r = run_slice(slicer, inputs, name, f"block{bidx}.log")
        results.append(r)
        bidx += 1
        _print_row(r)

    # ── DLA base: base_layer alone ────────────────────────────────────────────
    _run(_SliceBaseLayer(model), "base_layer")

    # ── DLA base: incremental levels ─────────────────────────────────────────
    for lvl in range(6):
        _run(_SliceBaseUpToLevel(model, lvl), f"base → level{lvl}")

    # ── DLAUp: add one IDA block at a time ────────────────────────────────────
    n_dla_idas = sum(
        1 for nm, _ in model.dla_up.named_children() if nm.startswith("ida_")
    )
    for n in range(1, n_dla_idas + 1):
        _run(_SliceDLAUpPartial(model, n), f"DLAUp ida_0..ida_{n-1}")

    # ── IDAUp: one step at a time ─────────────────────────────────────────────
    n_ida_steps = sum(
        1 for nm, _ in model.ida_up.named_children() if nm.startswith("proj_")
    )
    for s in range(1, n_ida_steps + 1):
        _run(_SliceIDAUpPartial(model, s), f"IDAUp step 1..{s}")

    # ── Heads ─────────────────────────────────────────────────────────────────
    for head_name in ["hm", "wh", "reg"]:
        _run(_SliceWithHead(model, head_name), f"backbone + {head_name} head")

    return results


def head_op_level_analysis(model, inputs, head_name: str) -> List[Dict]:
    """Sub-op PCC analysis inside a detection head (conv1 → relu → conv2)."""
    main_log.info("\n" + "=" * 62)
    main_log.info(f"PHASE 3 — Sub-op PCC analysis inside '{head_name}' head")
    main_log.info("=" * 62)

    results = []
    for sub_op in _SliceHeadSubOp._SUB_OPS:
        name = f"{head_name}_head.{sub_op}"
        log_file = f"head_{head_name}_{sub_op}.log"
        try:
            slicer = _SliceHeadSubOp(model, head_name, sub_op)
            r = run_slice(slicer, inputs, name, log_file)
        except Exception:
            main_log.error(f"  ERROR building slicer for {name}:\n{traceback.format_exc()}")
            r = {"block": name, "pcc": None, "log_file": log_file,
                 "cpu_stats": None, "tt_stats": None}
        results.append(r)
        _print_row(r)
    return results


def op_level_analysis(model, inputs, failing_block: Optional[str]) -> List[Dict]:
    """Fine-grained op-level analysis inside each IDAUp step."""
    main_log.info("\n" + "=" * 62)
    main_log.info("PHASE 2 — Op-level PCC analysis inside IDAUp")
    if failing_block:
        main_log.info(f"(first failing block: {failing_block})")
    main_log.info("=" * 62)

    results = []
    oidx = 1
    n_steps = sum(
        1 for nm, _ in model.ida_up.named_children() if nm.startswith("proj_")
    )

    for step in range(1, n_steps + 1):
        for sub_op in _SliceIDAUpSubOp._SUB_OPS:
            name = f"IDAUp.step{step}.{sub_op}"
            try:
                slicer = _SliceIDAUpSubOp(model, step, sub_op)
                r = run_slice(slicer, inputs, name, f"op{oidx}.log")
            except Exception:
                main_log.error(f"  ERROR building slicer for {name}:\n{traceback.format_exc()}")
                r = {"block": name, "pcc": None, "log_file": f"op{oidx}.log",
                     "cpu_stats": None, "tt_stats": None}
            results.append(r)
            oidx += 1
            _print_row(r)

    return results


# ── Pretty printing ───────────────────────────────────────────────────────────


def _print_row(r: Dict) -> None:
    pcc = r["pcc"]
    pcc_s = f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "N/A"
    ok = isinstance(pcc, float) and pcc == pcc and pcc >= 0.97
    flag = "OK  " if ok else "FAIL"
    main_log.info(f"  [{flag}]  {r['block']:<52}  PCC={pcc_s}  ({r['log_file']})")


def _first_fail(results: List[Dict]) -> Optional[str]:
    for r in results:
        pcc = r["pcc"]
        if isinstance(pcc, float) and (pcc != pcc or pcc < 0.97):  # nan or < 0.97
            return r["block"]
    return None


def print_final_report(
    block_results: List[Dict],
    op_results: List[Dict],
    head_op_results: Optional[List[Dict]] = None,
) -> None:
    sep = "=" * 72

    main_log.info("\n" + sep)
    main_log.info("BLOCK-LEVEL PCC TABLE")
    main_log.info(sep)
    main_log.info(f"  {'Status':<6}  {'Block':<54}  {'PCC':>10}  Log")
    main_log.info("  " + "-" * 68)
    for r in block_results:
        pcc = r["pcc"]
        pcc_s = f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "N/A"
        ok = isinstance(pcc, float) and pcc == pcc and pcc >= 0.97
        status = "OK  " if ok else "FAIL"
        main_log.info(f"  [{status}]  {r['block']:<54}  {pcc_s:>10}  {r['log_file']}")

    if op_results:
        main_log.info("\n" + sep)
        main_log.info("OPERATION-LEVEL PCC TABLE (IDAUp sub-ops)")
        main_log.info(sep)
        main_log.info(f"  {'Status':<6}  {'Op':<54}  {'PCC':>10}  Log")
        main_log.info("  " + "-" * 68)
        for r in op_results:
            pcc = r["pcc"]
            pcc_s = f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "N/A"
            ok = isinstance(pcc, float) and pcc == pcc and pcc >= 0.97
            status = "OK  " if ok else "FAIL"
            main_log.info(f"  [{status}]  {r['block']:<54}  {pcc_s:>10}  {r['log_file']}")

    if head_op_results:
        main_log.info("\n" + sep)
        main_log.info("DETECTION HEAD SUB-OP PCC TABLE")
        main_log.info(sep)
        main_log.info(f"  {'Status':<6}  {'Op':<54}  {'PCC':>10}  Log")
        main_log.info("  " + "-" * 68)
        for r in head_op_results:
            pcc = r["pcc"]
            pcc_s = f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "N/A"
            ok = isinstance(pcc, float) and pcc == pcc and pcc >= 0.97
            status = "OK  " if ok else "FAIL"
            # Print cpu and tt stats side by side for quick inspection
            cpu_s = r.get("cpu_stats") or {}
            tt_s  = r.get("tt_stats")  or {}
            extra = ""
            if cpu_s and tt_s:
                extra = (
                    f"  CPU std={cpu_s.get('std',0):.4f} max={cpu_s.get('max',0):.4f}"
                    f"  TT std={tt_s.get('std',0):.4f} max={tt_s.get('max',0):.4f}"
                )
            main_log.info(f"  [{status}]  {r['block']:<54}  {pcc_s:>10}  {r['log_file']}{extra}")

    failing_block = _first_fail(block_results)
    failing_op = _first_fail(op_results) if op_results else None
    failing_head_op = _first_fail(head_op_results) if head_op_results else None

    main_log.info("\n" + sep)
    main_log.info(f"  FIRST FAILING BLOCK     : {failing_block   or '— all blocks passed —'}")
    main_log.info(f"  FIRST FAILING IDAUP OP  : {failing_op      or '— all IDAUp ops passed —'}")
    main_log.info(f"  FIRST FAILING HEAD OP   : {failing_head_op or '— all head ops passed —'}")
    main_log.info(sep)
    main_log.info("  SUSPECTED ROOT CAUSE:")
    main_log.info("    The 'reg' head has a very small output dynamic range (std≈0.095)")
    main_log.info("    compared to 'hm' and 'wh' (std≈33).  Even small absolute errors")
    main_log.info("    introduced by TT hardware (bfloat16 math) translate to a large")
    main_log.info("    PCC degradation when the signal variance is this low.")
    main_log.info("    The sub-op PCC table above pinpoints exactly where in the head")
    main_log.info("    the precision loss begins (conv1, relu, or conv2).")
    main_log.info("    Fix candidates:")
    main_log.info("      1. Raise PCC threshold for 'reg' output specifically")
    main_log.info("      2. Keep 'reg' head in float32 (dtype override on that layer)")
    main_log.info("      3. Investigate TT compiler lowering of the failing conv op")
    main_log.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def main():
    main_log.info("=" * 72)
    main_log.info("CenterNet DLA-1x PCC Debug Script")
    main_log.info(f"Log directory : {LOG_DIR}")
    main_log.info(f"TT available  : {TT_AVAILABLE}")
    main_log.info("=" * 72)

    # ── load model + inputs ───────────────────────────────────────────────────
    main_log.info("Loading CenterNet DLA-1x model …")
    from third_party.tt_forge_models.centernet.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    loader = ModelLoader(variant=ModelVariant.DLA_1X_COCO)
    model = loader.load_model()
    inputs = loader.load_inputs()
    model.eval()
    main_log.info(f"Model loaded.  Input shape: {inputs.shape}  dtype: {inputs.dtype}")
    main_log.info(
        f"Model params : {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M"
    )
    main_log.info(f"first_level={model.first_level}  last_level={model.last_level}")

    # ── Phase 1: block-level ──────────────────────────────────────────────────
    block_results = block_level_analysis(model, inputs)
    failing_block = _first_fail(block_results)
    main_log.info(f"\nFirst failing block → {failing_block or 'None (all OK)'}")

    # ── Phase 2: op-level (IDAUp sub-ops) ────────────────────────────────────
    op_results = op_level_analysis(model, inputs, failing_block)

    # ── Phase 3: head sub-op analysis for each failing head ───────────────────
    # Always run all three heads so we can compare passing vs failing ones.
    head_op_results: List[Dict] = []
    for head_name in ["hm", "wh", "reg"]:
        head_op_results.extend(head_op_level_analysis(model, inputs, head_name))

    # ── Final report ──────────────────────────────────────────────────────────
    print_final_report(block_results, op_results, head_op_results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "failing_block": failing_block,
        "failing_op": _first_fail(op_results),
        "failing_head_op": _first_fail(head_op_results),
        "block_results": block_results,
        "op_results": op_results,
        "head_op_results": head_op_results,
    }
    json_path = LOG_DIR / "pcc_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    main_log.info(f"\nJSON summary saved → {json_path}")
    main_log.info("Done.")


if __name__ == "__main__":
    main()
