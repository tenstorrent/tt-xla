# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared ``run_op_test`` helpers with optional PCC / diff reporting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Sequence

import torch
import torch.nn as nn
from infra import Framework, run_op_test
from infra.evaluators import ComparisonConfig, TorchComparisonEvaluator
from infra.utilities import PyTree
from infra.workloads.torch_workload import TorchWorkload
from tests.infra.testers.single_chip.op.op_tester import OpTester
from torch.utils._pytree import tree_flatten


def print_metrics_enabled() -> bool:
    """Set ``JANUS_DECODER_PRINT_METRICS=1`` to print PCC and diff stats after each op test."""
    return os.environ.get("JANUS_DECODER_PRINT_METRICS", "0") == "1"


@dataclass(frozen=True)
class TensorMatchMetrics:
    pcc: float
    max_abs_diff: float
    mean_abs_diff: float
    rel_l2_diff: float


def _tensor_match_metrics(
    evaluator: TorchComparisonEvaluator,
    device_tensor: torch.Tensor,
    golden_tensor: torch.Tensor,
) -> TensorMatchMetrics:
    tt = device_tensor.detach().cpu().to(torch.float64)
    ref = golden_tensor.detach().cpu().to(torch.float64)
    diff = tt - ref
    abs_diff = diff.abs()
    ref_norm = ref.norm().item()
    rel_l2 = float(diff.norm().item() / ref_norm) if ref_norm > 0 else float("inf")
    return TensorMatchMetrics(
        pcc=float(
            evaluator._compare_pcc(
                tt,
                ref,
                evaluator._comparison_config.pcc,
                None,
            )
        ),
        max_abs_diff=float(abs_diff.max().item()),
        mean_abs_diff=float(abs_diff.mean().item()),
        rel_l2_diff=rel_l2,
    )


def _metrics_for_pytree(
    evaluator: TorchComparisonEvaluator,
    device_output: PyTree,
    golden_output: PyTree,
    prefix: str,
) -> list[tuple[str, TensorMatchMetrics]]:
    rows: list[tuple[str, TensorMatchMetrics]] = []

    def _collect(path: str, device_leaf: Any, golden_leaf: Any) -> None:
        if not isinstance(device_leaf, torch.Tensor):
            return
        rows.append(
            (
                path,
                _tensor_match_metrics(evaluator, device_leaf, golden_leaf),
            )
        )

    def _map_with_path(path: str, device_leaf: Any, golden_leaf: Any) -> None:
        if isinstance(device_leaf, torch.Tensor):
            _collect(path or "output", device_leaf, golden_leaf)
            return device_leaf
        if isinstance(device_leaf, (tuple, list)):
            for i, (d, g) in enumerate(zip(device_leaf, golden_leaf)):
                _map_with_path(f"{path}[{i}]" if path else f"[{i}]", d, g)
        return device_leaf

    if isinstance(device_output, torch.Tensor):
        rows.append(
            (
                prefix,
                _tensor_match_metrics(evaluator, device_output, golden_output),
            )
        )
    elif isinstance(device_output, (tuple, list)):
        for i, (d, g) in enumerate(zip(device_output, golden_output)):
            name = f"{prefix}[{i}]" if prefix else f"output[{i}]"
            if isinstance(d, torch.Tensor):
                rows.append((name, _tensor_match_metrics(evaluator, d, g)))
    else:
        flat_d, _ = tree_flatten(device_output)
        flat_g, _ = tree_flatten(golden_output)
        for i, (d, g) in enumerate(zip(flat_d, flat_g)):
            if isinstance(d, torch.Tensor) and isinstance(g, torch.Tensor):
                rows.append(
                    (
                        f"{prefix}[{i}]" if prefix else f"output[{i}]",
                        _tensor_match_metrics(evaluator, d, g),
                    )
                )

    return rows


def print_tensor_match_metrics(label: str, rows: Sequence[tuple[str, TensorMatchMetrics]]) -> None:
    print(f"\n--- {label} ---")
    print(
        f"{'tensor':<24}  {'pcc':>10}  {'max_abs':>12}  {'mean_abs':>12}  {'rel_l2':>10}"
    )
    print("-" * 76)
    for name, row in rows:
        print(
            f"{name:<24}  {row.pcc:>10.6f}  {row.max_abs_diff:>12.6e}  "
            f"{row.mean_abs_diff:>12.6e}  {row.rel_l2_diff:>10.6e}"
        )
    print("-" * 76)


def _split_stage_outputs(output: Any, num_stages: int) -> list[torch.Tensor]:
    """``[N, ...]`` stacked tensor or ``tuple``/``list`` of per-stage tensors."""
    if isinstance(output, (tuple, list)):
        if len(output) != num_stages:
            raise ValueError(
                f"Expected {num_stages} stage tensors, got {len(output)}"
            )
        return list(output)
    if isinstance(output, torch.Tensor):
        if output.shape[0] != num_stages:
            raise ValueError(
                f"Expected {num_stages} stages, got shape {tuple(output.shape)}"
            )
        return [output[index] for index in range(num_stages)]
    raise TypeError(f"Unsupported stage output type: {type(output)}")


def run_decoder_stacked_stage_profile_op_test(
    label: str,
    wrapper: nn.Module,
    inputs: Sequence[torch.Tensor],
    stage_names: Sequence[str],
    *,
    assert_on_failure: bool = True,
    return_metrics: bool = False,
) -> list[tuple[str, TensorMatchMetrics]] | None:
    """
    Run op test on stacked stage outputs; print PCC / diff per stage.

    Wrapper may return ``torch.stack`` of same-shaped stages or a ``tuple`` when shapes
    differ (e.g. attention Q/K/V vs hidden states).

    Asserts only on the final stage (``stage_names[-1]``) when ``assert_on_failure`` is True.
    """
    if len(stage_names) == 0:
        raise ValueError("stage_names must be non-empty")

    num_stages = len(stage_names)
    comparison_config = ComparisonConfig(assert_on_failure=False)
    evaluator = TorchComparisonEvaluator(comparison_config)
    metrics_rows: list[tuple[str, TensorMatchMetrics]] = []

    def _stage_comparator(
        device_output: Any,
        cpu_output: Any,
        _args: Any,
        _kwargs: Any,
    ) -> None:
        device_stages = _split_stage_outputs(device_output, num_stages)
        cpu_stages = _split_stage_outputs(cpu_output, num_stages)
        for index, name in enumerate(stage_names):
            metrics_rows.append(
                (
                    name,
                    _tensor_match_metrics(
                        evaluator,
                        device_stages[index],
                        cpu_stages[index],
                    ),
                )
            )
        if assert_on_failure:
            evaluator.evaluate(device_stages[-1], cpu_stages[-1])

    run_op_test(
        wrapper,
        list(inputs),
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        custom_comparator=_stage_comparator,
    )
    print_tensor_match_metrics(label, metrics_rows)
    if return_metrics:
        return list(metrics_rows)
    return None


def run_decoder_stacked_stage_forge_vs_cpu_isolated(
    label: str,
    build_model_and_inputs: Callable[[], tuple[nn.Module, list[torch.Tensor]]],
    stage_names: Sequence[str],
    *,
    assert_on_failure: bool = True,
    return_metrics: bool = False,
) -> list[tuple[str, TensorMatchMetrics]] | None:
    """
    Forge vs CPU with **separate** module instances (fair PCC).

    ``run_op_test`` reuses one module: CPU forward mutates ``past_key_values``, then
    Forge runs on dirty state and can show a false ~0.77 on ``self_attn``. This helper
    builds fresh ``(wrapper, inputs)`` for CPU and again for Forge.
    """
    if len(stage_names) == 0:
        raise ValueError("stage_names must be non-empty")

    num_stages = len(stage_names)
    comparison_config = ComparisonConfig(assert_on_failure=False)
    evaluator = TorchComparisonEvaluator(comparison_config)
    tester = OpTester(
        comparison_config=comparison_config,
        framework=Framework.TORCH,
    )

    cpu_wrapper, cpu_inputs = build_model_and_inputs()
    cpu_out = tester._device_runner.run_on_cpu(
        TorchWorkload(model=cpu_wrapper, args=cpu_inputs)
    )

    forge_wrapper, forge_inputs = build_model_and_inputs()
    forge_workload = TorchWorkload(model=forge_wrapper, args=forge_inputs)
    tester._compile_for_tt_device(forge_workload)
    forge_out = tester._device_runner.run_on_tt_device(forge_workload)

    cpu_stages = _split_stage_outputs(cpu_out, num_stages)
    forge_stages = _split_stage_outputs(forge_out, num_stages)
    metrics_rows: list[tuple[str, TensorMatchMetrics]] = []
    for index, name in enumerate(stage_names):
        metrics_rows.append(
            (
                name,
                _tensor_match_metrics(
                    evaluator,
                    forge_stages[index],
                    cpu_stages[index],
                ),
            )
        )
    if assert_on_failure:
        evaluator.evaluate(forge_stages[-1], cpu_stages[-1])

    print_tensor_match_metrics(label, metrics_rows)
    if return_metrics:
        return list(metrics_rows)
    return None


def run_forge_vs_cpu_op_test_isolated(
    build_model_and_inputs: Callable[[], tuple[nn.Module, list[torch.Tensor]]],
    *,
    label: str = "forge_vs_cpu_isolated",
    assert_on_failure: bool = True,
    custom_comparator: Callable[..., None] | None = None,
) -> None:
    """
    Forge vs CPU with separate module instances (fair single-output PCC).

    Use instead of ``run_op_test`` when ``past_key_values`` mutates across forwards.
    """
    comparison_config = ComparisonConfig(assert_on_failure=False)
    tester = OpTester(
        comparison_config=comparison_config,
        framework=Framework.TORCH,
    )

    cpu_wrapper, cpu_inputs = build_model_and_inputs()
    cpu_out = tester._device_runner.run_on_cpu(
        TorchWorkload(model=cpu_wrapper, args=cpu_inputs)
    )

    forge_wrapper, forge_inputs = build_model_and_inputs()
    forge_workload = TorchWorkload(model=forge_wrapper, args=forge_inputs)
    tester._compile_for_tt_device(forge_workload)
    forge_out = tester._device_runner.run_on_tt_device(forge_workload)

    if custom_comparator is not None:
        custom_comparator(forge_out, cpu_out, forge_inputs, {})
    else:
        evaluator = TorchComparisonEvaluator(comparison_config)
        metrics = _tensor_match_metrics(evaluator, forge_out, cpu_out)
        print_tensor_match_metrics(label, [("output", metrics)])
        if assert_on_failure:
            evaluator.evaluate(forge_out, cpu_out)


def run_decoder_op_test_collect_metrics(
    wrapper: nn.Module,
    inputs: Sequence[torch.Tensor],
) -> TensorMatchMetrics:
    """Run op test vs CPU golden; return metrics without asserting."""
    comparison_config = ComparisonConfig(assert_on_failure=False)
    evaluator = TorchComparisonEvaluator(comparison_config)
    collected: list[TensorMatchMetrics] = []

    def _collect_comparator(
        device_output: PyTree,
        golden_output: PyTree,
        _args: Any,
        _kwargs: Any,
    ) -> None:
        rows = _metrics_for_pytree(evaluator, device_output, golden_output, prefix="")
        if not rows:
            raise RuntimeError("No tensor outputs to compare")
        collected.append(rows[0][1])

    run_op_test(
        wrapper,
        list(inputs),
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        custom_comparator=_collect_comparator,
    )
    return collected[0]


def run_decoder_op_test(
    label: str,
    wrapper: nn.Module,
    inputs: Sequence[torch.Tensor],
    *,
    assert_on_failure: bool = True,
    comparison_config: ComparisonConfig | None = None,
) -> None:
    """
    Run ``run_op_test``; optionally print PCC / diff metrics (``JANUS_DECODER_PRINT_METRICS=1``).
    """
    if comparison_config is None:
        comparison_config = ComparisonConfig(assert_on_failure=assert_on_failure)

    if not print_metrics_enabled():
        run_op_test(
            wrapper,
            list(inputs),
            framework=Framework.TORCH,
            comparison_config=comparison_config,
        )
        return

    evaluator = TorchComparisonEvaluator(comparison_config)
    metrics_rows: list[tuple[str, TensorMatchMetrics]] = []

    def _metrics_comparator(
        device_output: PyTree,
        golden_output: PyTree,
        _args: Any,
        _kwargs: Any,
    ) -> None:
        metrics_rows.extend(
            _metrics_for_pytree(evaluator, device_output, golden_output, prefix="")
        )
        if assert_on_failure:
            evaluator.evaluate(device_output, golden_output)

    run_op_test(
        wrapper,
        list(inputs),
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        custom_comparator=_metrics_comparator,
    )
    print_tensor_match_metrics(label, metrics_rows)
