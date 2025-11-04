# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
import pytest
from flax import nnx
from infra import JaxModelTester, RunMode
from jaxtyping import PyTree

from .model import SimpleMLP

# ----- Tester -----


class SimpleMlpTester(JaxModelTester):
    """Simple MLP tester using standard NNX layers."""

    # @override
    def _get_model(self) -> nnx.Module:
        rngs = nnx.Rngs(0)
        return SimpleMLP(rngs)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        batch_size = 32
        input_dim = 64
        act_shape = (batch_size, input_dim)
        act = jax.numpy.ones(act_shape)
        return [act]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> SimpleMlpTester:
    return SimpleMlpTester()


@pytest.fixture
def training_tester() -> SimpleMlpTester:
    return SimpleMlpTester(run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
def test_simple_mlp_inference(inference_tester: SimpleMlpTester):
    inference_tester.test()


@pytest.mark.push
def test_simple_mlp_training(training_tester: SimpleMlpTester):
    training_tester.test()
