# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import pytest
from flax import nnx
from infra import JaxModelTester, RunMode
from jaxtyping import PyTree

from ..model import ExampleModel

# ----- Tester -----


class ExampleModelOnlyKwargsTester(JaxModelTester):
    """
    Example tester showcasing how to use only keyword arguments for model's forward
    method.
    """

    # @override
    def _get_model(self) -> nnx.Module:
        return ExampleModel()

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        act_shape = (32, 784)
        act = jax.numpy.ones(act_shape)
        return [act]

    # @override
    def _get_input_parameters(self) -> PyTree:
        return ()

    # @override
    def _get_forward_method_name(self) -> str:
        return "__call__"

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
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
        return {"act": input_activation, "w0": w0, "b0": b0, "w1": w1, "b1": b1}


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ExampleModelOnlyKwargsTester:
    return ExampleModelOnlyKwargsTester()


@pytest.fixture
def training_tester() -> ExampleModelOnlyKwargsTester:
    return ExampleModelOnlyKwargsTester(run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
def test_example_model_inference(inference_tester: ExampleModelOnlyKwargsTester):
    inference_tester.test()


@pytest.mark.push
def test_example_model_training(training_tester: ExampleModelOnlyKwargsTester):
    training_tester.test()
