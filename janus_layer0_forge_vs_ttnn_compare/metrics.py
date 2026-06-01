# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC / diff metrics (matches tt-xla ``TorchComparisonEvaluator`` style)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

STAGE_NAMES: tuple[str, ...] = ("decode_setup", "input_layernorm", "self_attn")


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


def split_stacked(stacked: torch.Tensor, num_stages: int = 3) -> list[torch.Tensor]:
    if stacked.ndim != 4 or stacked.shape[0] != num_stages:
        raise ValueError(f"Expected [{num_stages}, B, S, H], got {tuple(stacked.shape)}")
    return [stacked[index] for index in range(num_stages)]


def print_comparison_table(
    title: str,
    reference_label: str,
    actual_label: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> list[tuple[str, StagePccMetrics]]:
    ref_stages = split_stacked(reference)
    act_stages = split_stacked(actual)
    rows: list[tuple[str, StagePccMetrics]] = []
    print(f"\n--- {title}: {actual_label} vs {reference_label} ---")
    print(f"{'stage':<28} {'pcc':>10} {'max_abs':>12} {'mean_abs':>12} {'rel_l2':>12}")
    print("-" * 76)
    for name, ref, act in zip(STAGE_NAMES, ref_stages, act_stages):
        m = stage_metrics(ref, act)
        rows.append((name, m))
        print(
            f"{name:<28} {m.pcc:10.6f} {m.max_abs_diff:12.6e} "
            f"{m.mean_abs_diff:12.6e} {m.rel_l2_diff:12.6e}"
        )
    print("-" * 76)
    return rows
