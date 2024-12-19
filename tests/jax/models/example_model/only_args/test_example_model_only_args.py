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


class ExampleModelOnlyArgsTester(ModelTester):
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
        input_activations = self._get_input_activations()
        assert len(input_activations) == 1
        input_activation = input_activations[0]

        # Mix activations, weights and biases to match forward method signature.
        return [input_activation, w0, b0, w1, b1]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ExampleModelOnlyArgsTester:
    return ExampleModelOnlyArgsTester()


@pytest.fixture
def training_tester() -> ExampleModelOnlyArgsTester:
    return ExampleModelOnlyArgsTester(RunMode.TRAINING)


# ----- Tests -----


def test_example_model_inference(inference_tester: ExampleModelOnlyArgsTester):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_example_model_training(training_tester: ExampleModelOnlyArgsTester):
    training_tester.test()
