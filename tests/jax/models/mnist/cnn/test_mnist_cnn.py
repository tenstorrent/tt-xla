# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from infra import ModelTester, RunMode

from tests.jax.models.mnist.cnn.model_implementation import MNISTCNNModel


class MNISTCNNTester(ModelTester):
    """Tester for MNIST CNN model."""

    # @override
    def _get_model(self) -> nn.Module:
        return MNISTCNNModel()

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        img = jnp.ones((4, 28, 28, 1))  # B, H, W, C
        # Channels is 1 as MNIST is in grayscale.
        return img

    # @override
    def _get_forward_method_args(self):
        inp = self._get_input_activations()

        # Example of flax.linen convention of first instatiating a model object
        # and then later calling init to generate a set of initial tensors(parameters and maybe some extra state)
        parameters = self._model.init(jax.random.PRNGKey(42), inp, train=False)

        return [parameters, inp]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {"train": False}

    # @override
    def _get_static_argnames(self):
        return ["train"]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MNISTCNNTester:
    return MNISTCNNTester()


@pytest.fixture
def training_tester() -> MNISTCNNTester:
    return MNISTCNNTester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.skip(
    reason='void mlir::OperationConverter::finalize(mlir::ConversionPatternRewriter &): Assertion `newValue && "replacement value not found"\' failed.'
)  # This is a segfault, marking it as xfail would bring down the whole test suite
def test_mnist_inference(
    inference_tester: MNISTCNNTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_training(
    training_tester: MNISTCNNTester,
):
    training_tester.test()


if __name__ == "__main__":
    MNISTCNNTester().test()
