# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare TTNN ``graph_0`` output vs tt-xla CPU golden (Layer0LnAttnNoDep)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from cpu_reference.forward import run_forward_from_fixtures

STAGE_NAMES: tuple[str, ...] = ("decode_setup", "input_layernorm", "self_attn")

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_GOLDEN_FILE = _GOLDEN_DIR / "stacked_stages_pro_1b.pt"


@dataclass(frozen=True)
class StagePccMetrics:
    pcc: float
    max_abs_diff: float
    mean_abs_diff: float
    rel_l2_diff: float


def compute_pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    ref = expected.detach().cpu().to(torch.float64).flatten()
    got = actual.detach().cpu().to(torch.float64).flatten()
    if ref.numel() == 0:
        return 1.0
    ref = ref - ref.mean()
    got = got - got.mean()
    denom = (ref.norm() * got.norm()).clamp(min=1e-12)
    return float((ref * got).sum() / denom)


def stage_metrics(expected: torch.Tensor, actual: torch.Tensor) -> StagePccMetrics:
    ref = expected.detach().cpu().to(torch.float64)
    got = actual.detach().cpu().to(torch.float64)
    diff = got - ref
    abs_diff = diff.abs()
    ref_norm = ref.norm().item()
    rel_l2 = float(diff.norm().item() / ref_norm) if ref_norm > 0 else float("inf")
    return StagePccMetrics(
        pcc=compute_pcc(expected, actual),
        max_abs_diff=float(abs_diff.max().item()),
        mean_abs_diff=float(abs_diff.mean().item()),
        rel_l2_diff=rel_l2,
    )


def _ttnn_output_to_stacked(ttnn_outputs: Sequence[Any]) -> torch.Tensor:
    stacked = ttnn_outputs[0]
    import ttnn

    if hasattr(stacked, "cpu") and type(stacked).__module__.startswith("ttnn"):
        stacked = ttnn.to_torch(stacked)
    elif hasattr(stacked, "detach"):
        stacked = stacked.detach().cpu()
    if stacked.ndim != 4 or stacked.shape[0] != 3:
        raise ValueError(f"Expected [3, B, S, H], got {tuple(stacked.shape)}")
    return stacked


def load_or_compute_cpu_golden(
    *,
    variant: str = "Pro_1B",
    refresh: bool = False,
) -> torch.Tensor:
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    if not refresh and _GOLDEN_FILE.is_file():
        return torch.load(_GOLDEN_FILE, weights_only=True)
    golden = run_forward_from_fixtures(variant)
    torch.save(golden, _GOLDEN_FILE)
    print(f"Wrote CPU golden to {_GOLDEN_FILE}  shape={tuple(golden.shape)}")
    return golden


def compare_layer0_ln_attn_stages(
    ttnn_outputs: Sequence[Any],
    *,
    variant: str = "Pro_1B",
    refresh_cpu_golden: bool = False,
) -> list[StagePccMetrics]:
    """
    TTNN export vs CPU golden.

    CPU golden = ``janus_layer0_build.run_forward_stacked`` via tt-xla tests (no ``torch_xla``;
    same graph as codegen compare gate / no-dep sanity CPU side).

    Expect **~0.99** on ``self_attn`` here (TTNN vs CPU). The **~0.77** drop is **Forge vs this
    same CPU** on tt-xla only — not TTNN vs CPU.
    """
    cpu_stacked = load_or_compute_cpu_golden(variant=variant, refresh=refresh_cpu_golden)
    tt_stacked = _ttnn_output_to_stacked(ttnn_outputs)

    if tuple(cpu_stacked.shape) != tuple(tt_stacked.shape):
        raise ValueError(
            f"Shape mismatch: CPU {tuple(cpu_stacked.shape)} vs TT {tuple(tt_stacked.shape)}"
        )

    metrics: list[StagePccMetrics] = []
    print(f"\n--- layer0_ln_attn TTNN vs CPU (Layer0LnAttnNoDep, {variant}) ---")
    print(f"{'stage':<28} {'pcc':>10} {'max_abs':>12} {'mean_abs':>12} {'rel_l2':>12}")
    print("-" * 76)
    for index, name in enumerate(STAGE_NAMES):
        m = stage_metrics(cpu_stacked[index], tt_stacked[index])
        metrics.append(m)
        print(
            f"{name:<28} {m.pcc:10.6f} {m.max_abs_diff:12.6e} "
            f"{m.mean_abs_diff:12.6e} {m.rel_l2_diff:12.6e}"
        )
    print("-" * 76)

    attn_pcc = metrics[2].pcc
    print(
        f"\nself_attn PCC={attn_pcc:.4f}  (TTNN vs CPU golden)")
    print(
        "CPU golden = same build as codegen/compare gate (expect ~0.99 on self_attn). "
        "Forge vs this CPU on tt-xla ≈ 0.77 — see janus_layer0_forge_vs_ttnn_compare/EXPERIMENTS.md"
    )
    return metrics
