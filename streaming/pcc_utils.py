"""Per-layer activation capture / compare utilities.

Two operation modes (controlled by env var STREAM_REF_MODE):

  - capture: every layer's output activation is dumped to disk after
    the forward call. Use this on a known-good run to build a golden
    reference set.

  - compare: the previously captured reference is loaded and a PCC
    (Pearson correlation coefficient) is computed against the current
    output. Useful for validating that a code/runtime change preserves
    numerics.

  - none / unset: no-op (default).

The reference directory defaults to /tmp/stream_ref.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


REF_MODE = os.environ.get("STREAM_REF_MODE", "none").lower()
REF_DIR = Path(os.environ.get("STREAM_REF_DIR", "/tmp/stream_ref"))

# inline mode: CPU eager forward run inside the device loop, compared
# immediately. No file I/O. CPU result is discarded after PCC.
INLINE_PCC = bool(int(os.environ.get("STREAM_INLINE_PCC", "0")))

if REF_MODE == "capture":
    REF_DIR.mkdir(parents=True, exist_ok=True)


def _ref_path(step: int, layer_id: int, tag: str = "out") -> Path:
    return REF_DIR / f"s{step:02d}_l{layer_id:02d}_{tag}.pt"


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient on flattened tensors.

    Mathematically PCC ∈ [-1, 1]. Two pragmatic guards on top of the raw
    formula so the returned value is meaningful in float32 arithmetic:

    1. NaN/inf in either input → return NaN (caller halts).
    2. allclose(a, b) → return 1.0 (the formula is ill-conditioned for
       near-identical large tensors; it can produce values like 1.004
       purely from accumulation order between numerator and denominator
       reductions, which is misleading). Mirrors the framework's own
       PCC implementation in tests/infra/evaluators/torch_comparison_evaluator.py.
    3. Final clamp to [-1, 1] for any residual fp32 overshoot.
    """
    a = a.flatten().float()
    b = b.flatten().float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return float("nan")
    if torch.allclose(a, b, rtol=1e-5, atol=1e-8):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    v = float((a * b).sum() / denom)
    return max(-1.0, min(1.0, v))


def shape_match(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype


def inline_pcc(
    step: int,
    layer_id: int,
    cpu_tensor: torch.Tensor,
    device_tensor: torch.Tensor,
    *,
    tag: str = "out",
) -> float:
    """Compare a CPU eager output tensor against a device output tensor
    in-line (no file I/O). Returns PCC."""
    a = cpu_tensor.detach().to("cpu").float()
    b = device_tensor.detach().to("cpu").float()
    if not shape_match(cpu_tensor.detach().to("cpu"), device_tensor.detach().to("cpu")):
        print(
            f"[pcc] shape/dtype mismatch s{step} l{layer_id} {tag}: "
            f"cpu {tuple(a.shape)}/{a.dtype} vs dev {tuple(b.shape)}/{b.dtype}",
            flush=True,
        )
        return float("nan")
    v = pcc(a, b)
    print(f"[pcc] s{step:02d} l{layer_id:02d} {tag} pcc={v:.6f}", flush=True)
    return v


def capture_or_compare(
    step: int,
    layer_id: int,
    tensor: torch.Tensor,
    *,
    tag: str = "out",
) -> Optional[float]:
    """Capture or compare a single activation. Returns PCC if compared."""
    if REF_MODE == "none":
        return None
    path = _ref_path(step, layer_id, tag)

    # Always materialize to host CPU (caller may pass an XLA tensor).
    if tensor.device.type != "cpu":
        cpu = tensor.detach().to("cpu")
    else:
        cpu = tensor.detach()
    cpu = cpu.float() if cpu.is_floating_point() else cpu

    if REF_MODE == "capture":
        torch.save(cpu, path)
        return None
    elif REF_MODE == "compare":
        if not path.exists():
            print(f"[pcc] missing reference {path}", flush=True)
            return None
        ref = torch.load(path, map_location="cpu", weights_only=False)
        if not shape_match(cpu, ref):
            print(
                f"[pcc] shape/dtype mismatch s{step} l{layer_id} {tag}: "
                f"got {tuple(cpu.shape)}/{cpu.dtype} vs ref {tuple(ref.shape)}/{ref.dtype}",
                flush=True,
            )
            return None
        v = pcc(cpu, ref)
        print(f"[pcc] s{step:02d} l{layer_id:02d} {tag} pcc={v:.6f}", flush=True)
        return v
    return None
