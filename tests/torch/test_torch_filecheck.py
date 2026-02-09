# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch FileCheck tests for validating generated MLIR patterns."""

from typing import Any, Dict, Sequence

import pytest
import torch
from infra import (
    Framework,
    run_graph_test,
    run_graph_test_with_random_inputs,
    run_op_test,
    run_op_test_with_random_inputs,
)

from tests.infra import ComparisonConfig, Model, RunMode, TorchModelTester
from tests.infra.testers.compiler_config import CompilerConfig


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.filecheck(["add.ttnn.mlir"])
@pytest.mark.parametrize("random_inputs", [True, False])
@pytest.mark.parametrize("test_infra", ["op", "graph"])
def test_op_graph_filecheck(test_infra, random_inputs, request):
    """Test filecheck with Torch op and graph testers."""

    class Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    add = Add()

    if test_infra == "op":
        if random_inputs:
            run_op_test_with_random_inputs(
                add,
                [(32, 32), (32, 32)],
                framework=Framework.TORCH,
                request=request,
            )
        else:
            run_op_test(
                add,
                [torch.randn(32, 32), torch.randn(32, 32)],
                framework=Framework.TORCH,
                request=request,
            )
    else:
        if random_inputs:
            run_graph_test_with_random_inputs(
                add,
                [(32, 32), (32, 32)],
                framework=Framework.TORCH,
                request=request,
            )
        else:
            run_graph_test(
                add,
                [torch.randn(32, 32), torch.randn(32, 32)],
                framework=Framework.TORCH,
                request=request,
            )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.filecheck(["add.ttnn.mlir"])
def test_model_filecheck(request):
    """Test filecheck with Torch model tester."""

    class SimpleLinearModel(torch.nn.Module):
        """Lightweight fake model for testing filecheck infrastructure."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(x) + x

    class SimpleLinearModelTester(TorchModelTester):
        """Tester for simple linear model."""

        def __init__(
            self,
            comparison_config: ComparisonConfig = ComparisonConfig(),
            run_mode: RunMode = RunMode.INFERENCE,
            compiler_config: CompilerConfig = None,
            dtype_override=None,
        ) -> None:
            self._model_instance = SimpleLinearModel()
            self._inputs = [torch.randn(32, 32, dtype=torch.bfloat16)]
            super().__init__(
                comparison_config,
                run_mode,
                compiler_config,
                dtype_override=dtype_override,
            )

        def _get_model(self) -> Model:
            return self._model_instance

        def _get_input_activations(self) -> Dict | Sequence[Any]:
            return self._inputs

    tester = SimpleLinearModelTester(
        comparison_config=ComparisonConfig(),
        run_mode=RunMode.INFERENCE,
        compiler_config=None,
        dtype_override=None,
    )
    tester.test(request=request)
