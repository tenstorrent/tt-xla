# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for custom comparator in the test infrastructure.

Verifies that:
- When custom_comparator is provided: OpTester, GraphTester, and TorchModelTester use that
  instead of the default evaluator.
- When no custom_comparator is provided: OpTester, GraphTester, and TorchModelTester use
  the default evaluator.
"""

from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
from infra import Framework, TorchModelTester
from infra.workloads import TorchWorkload

from tests.infra.testers.single_chip.graph.graph_tester import GraphTester
from tests.infra.testers.single_chip.op.op_tester import OpTester

_DUMMY = torch.ones(32, 32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_runner():
    """Returns a MagicMock device runner whose run methods return _DUMMY."""
    runner = MagicMock()
    runner.run_on_cpu.return_value = _DUMMY
    runner.run_on_tt_device.return_value = _DUMMY
    return runner


def _make_torch_workload() -> TorchWorkload:
    """Builds a minimal Torch add workload (CPU-only tensors, no TT device)."""

    class _Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    return TorchWorkload(
        model=_Add(),
        args=[torch.randn(32, 32), torch.randn(32, 32)],
    )


class _SimpleTorchNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _SimpleTorchModelTester(TorchModelTester):
    def __init__(self, custom_comparator: Optional[Callable] = None):
        super().__init__(custom_comparator=custom_comparator)

    def _get_model(self) -> torch.nn.Module:
        return _SimpleTorchNN()

    def _get_input_activations(self) -> torch.Tensor:
        return torch.ones(8, 16)


# ---------------------------------------------------------------------------
# OpTester tests
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_op_tester_custom_comparator_is_called():
    """Positive: custom_comparator is invoked when provided to OpTester."""
    comparator = MagicMock()
    with patch(
        "tests.infra.testers.base_tester.DeviceRunnerFactory.create_runner",
        return_value=_make_mock_runner(),
    ):
        tester = OpTester(framework=Framework.TORCH, custom_comparator=comparator)

    with patch.object(tester, "_compile_for_tt_device"), patch(
        "tests.infra.testers.single_chip.op.op_tester.compile_torch_workload_for_cpu"
    ):
        tester.test(_make_torch_workload())

    comparator.assert_called_once()


@pytest.mark.push
def test_op_tester_custom_comparator_is_not_called():
    """Negative: default evaluator is used when no custom_comparator is provided."""
    with patch(
        "tests.infra.testers.base_tester.DeviceRunnerFactory.create_runner",
        return_value=_make_mock_runner(),
    ):
        tester = OpTester(framework=Framework.TORCH)

    assert tester._custom_comparator is None

    with patch.object(tester, "_compile_for_tt_device"), patch(
        "tests.infra.testers.single_chip.op.op_tester.compile_torch_workload_for_cpu"
    ), patch.object(tester._evaluator, "evaluate") as mock_evaluate:
        tester.test(_make_torch_workload())

    mock_evaluate.assert_called_once()


# ---------------------------------------------------------------------------
# GraphTester tests
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_graph_tester_custom_comparator_is_called():
    """Positive: custom_comparator is invoked when provided to GraphTester."""
    comparator = MagicMock()
    with patch(
        "tests.infra.testers.base_tester.DeviceRunnerFactory.create_runner",
        return_value=_make_mock_runner(),
    ):
        tester = GraphTester(framework=Framework.TORCH, custom_comparator=comparator)

    with patch.object(tester, "_compile_for_tt_device"), patch(
        "tests.infra.testers.single_chip.op.op_tester.compile_torch_workload_for_cpu"
    ):
        tester.test(_make_torch_workload())

    comparator.assert_called_once()


@pytest.mark.push
def test_graph_tester_custom_comparator_is_not_called():
    """Negative: default evaluator is used when no custom_comparator is provided."""
    with patch(
        "tests.infra.testers.base_tester.DeviceRunnerFactory.create_runner",
        return_value=_make_mock_runner(),
    ):
        tester = GraphTester(framework=Framework.TORCH)

    assert tester._custom_comparator is None

    with patch.object(tester, "_compile_for_tt_device"), patch(
        "tests.infra.testers.single_chip.op.op_tester.compile_torch_workload_for_cpu"
    ), patch.object(tester._evaluator, "evaluate") as mock_evaluate:
        tester.test(_make_torch_workload())

    mock_evaluate.assert_called_once()


# ---------------------------------------------------------------------------
# ModelTester tests
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_model_tester_custom_comparator_is_called():
    """Positive: custom_comparator is invoked when provided to ModelTester."""
    comparator = MagicMock()
    with patch(
        "tests.infra.testers.base_tester.DeviceRunnerFactory.create_runner",
        return_value=_make_mock_runner(),
    ):
        tester = _SimpleTorchModelTester(custom_comparator=comparator)

    with patch.object(tester, "_compile_for_cpu"), patch.object(
        tester, "_compile_for_tt_device"
    ):
        tester.test()

    comparator.assert_called_once()


@pytest.mark.push
def test_model_tester_custom_comparator_is_not_called():
    """Negative: default evaluator is used when no custom_comparator is provided."""
    with patch(
        "tests.infra.testers.base_tester.DeviceRunnerFactory.create_runner",
        return_value=_make_mock_runner(),
    ):
        tester = _SimpleTorchModelTester()

    assert tester._custom_comparator is None

    with patch.object(tester, "_compile_for_cpu"), patch.object(
        tester, "_compile_for_tt_device"
    ), patch.object(tester._evaluator, "evaluate") as mock_evaluate:
        tester.test()

    mock_evaluate.assert_called_once()
