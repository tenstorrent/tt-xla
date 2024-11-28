# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
import pytest
from flax import nnx
from infra import ComparisonConfig, ModelTester, RunMode

from ..model import ExampleModel

# ----- Tester -----


class ExampleModelTester(ModelTester):
    """
    Example tester showcasing how to use only positional arguments for model's forward
    method.
    """

    # @override
    @staticmethod
    def _get_model() -> nnx.Module:
        return ExampleModel()

    # @override
    @staticmethod
    def _get_input_activations() -> Sequence[jax.Array]:
        act_shape = (32, 784)
        act = jax.numpy.ones(act_shape)
        return [act]

    # @override
    @staticmethod
    def _get_forward_method_name() -> str:
        return "__call__"

    # @override
    def _get_forward_method_args(self) -> Sequence[jax.Array]:
        # Use stored `self._model` to fetch model attributes.
        # Asserts are just sanity checks, no need to use them every time.
        assert hasattr(self._model, "w0")
        assert hasattr(self._model, "w1")
        assert hasattr(self._model, "b0")
        assert hasattr(self._model, "b1")

        w0 = self._model.w0
        w1 = self._model.w1
        b0 = self._model.b0
        b1 = self._model.b1

        # Fetch activations.
        acts = self._get_input_activations()
        assert len(acts) == 1
        act = acts[0]

        # Mix activations, weights and biases to match forward method signature.
        return [act, w0, b0, w1, b1]


# ----- Fixtures -----


@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.atol.disable()
    return config


@pytest.fixture
def inference_tester(comparison_config: ComparisonConfig) -> ExampleModelTester:
    return ExampleModelTester(comparison_config)


@pytest.fixture
def training_tester(comparison_config: ComparisonConfig) -> ExampleModelTester:
    return ExampleModelTester(comparison_config, RunMode.TRAINING)


# ----- Tests -----


def test_example_model_inference(inference_tester: ExampleModelTester):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_example_model_training(training_tester: ExampleModelTester):
    training_tester.test()
