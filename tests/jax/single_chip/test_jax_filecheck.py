# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from infra import Framework
from infra.testers.single_chip.graph.graph_tester import (
    run_graph_test,
    run_graph_test_with_random_inputs,
)
from infra.testers.single_chip.op.op_tester import (
    run_op_test,
    run_op_test_with_random_inputs,
)

from tests.infra import ComparisonConfig, JaxModelTester
from tests.infra.testers.single_chip.model.model_tester import RunMode


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.filecheck(["add.ttnn.mlir"])
@pytest.mark.parametrize("random_inputs", [True, False])
@pytest.mark.parametrize("test_infra", ["op", "graph"])
def test_op_graph_filecheck(test_infra, random_inputs, request):
    """Test filecheck with JAX op and graph testers."""

    def add(x, y):
        return jnp.add(x, y)

    if test_infra == "op":
        if random_inputs:
            run_op_test_with_random_inputs(
                add,
                [(32, 32), (32, 32)],
                framework=Framework.JAX,
                request=request,
            )
        else:
            run_op_test(
                add,
                [jnp.ones((32, 32)), jnp.ones((32, 32))],
                framework=Framework.JAX,
                request=request,
            )
    else:
        if random_inputs:
            run_graph_test_with_random_inputs(
                add,
                [(32, 32), (32, 32)],
                framework=Framework.JAX,
                request=request,
            )
        else:
            run_graph_test(
                add,
                [jnp.ones((32, 32)), jnp.ones((32, 32))],
                framework=Framework.JAX,
                request=request,
            )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.filecheck(["add.ttnn.mlir"])
def test_model_filecheck(request):
    """Test filecheck with JAX model tester."""

    class SimpleLinearModel(nnx.Module):
        """Lightweight fake model for testing filecheck infrastructure."""

        def __init__(self, rngs: nnx.Rngs):
            self.linear = nnx.Linear(32, 32, use_bias=False, rngs=rngs)

        def __call__(self, x):
            return self.linear(x) + x

    class SimpleLinearModelTester(JaxModelTester):
        """Tester for simple linear model."""

        def __init__(
            self,
            comparison_config: ComparisonConfig = ComparisonConfig(),
            run_mode: RunMode = RunMode.INFERENCE,
        ) -> None:
            self._model_instance = SimpleLinearModel(rngs=nnx.Rngs(0))
            self._inputs = {"x": jnp.ones((32, 32))}
            super().__init__(comparison_config, run_mode)

        def _get_model(self) -> nnx.Module:
            return self._model_instance

        def _get_input_activations(self) -> Sequence[jax.Array]:
            return self._inputs

    tester = SimpleLinearModelTester(
        comparison_config=ComparisonConfig(),
        run_mode=RunMode.INFERENCE,
    )
    tester.test(request=request)
