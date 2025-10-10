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


class ExampleModelMixedArgsAndKwargsTester(JaxModelTester):
    """
    Example tester showcasing how to use both positional and keyword arguments for
    model's forward method.

    This is a completely artificial example. In most cases only one of
    {`_get_forward_method_args`, `_get_forward_method_kwargs`} will suffice.
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
    def _get_forward_method_args(self) -> Sequence[jax.Array]:
        """Returns just input activations as positional arg."""
        input_activations = self._get_input_activations()
        assert len(input_activations) == 1
        input_activation = input_activations[0]
        return [input_activation]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        """Returns weights and biases as keyword args."""
        assert hasattr(self._model, "w0")
        assert hasattr(self._model, "w1")
        assert hasattr(self._model, "b0")
        assert hasattr(self._model, "b1")

        w0 = self._model.w0
        w1 = self._model.w1
        b0 = self._model.b0
        b1 = self._model.b1

        # Order does not matter.
        return {"b1": b1, "w1": w1, "w0": w0, "b0": b0}


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ExampleModelMixedArgsAndKwargsTester:
    return ExampleModelMixedArgsAndKwargsTester()


@pytest.fixture
def training_tester() -> ExampleModelMixedArgsAndKwargsTester:
    return ExampleModelMixedArgsAndKwargsTester(run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
def test_example_model_inference(
    inference_tester: ExampleModelMixedArgsAndKwargsTester,
):
    inference_tester.test()


@pytest.mark.push
def test_example_model_training(training_tester: ExampleModelMixedArgsAndKwargsTester):
    training_tester.test()
