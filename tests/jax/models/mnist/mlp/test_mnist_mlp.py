# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
import pytest
from flax import linen as nn
from infra import ComparisonConfig, ModelTester, RunMode

from .model_implementation import MNISTMLPModel


class MNISTMLPTester(ModelTester):
    """Tester for MNIST MLP model."""

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._hidden_sizes = hidden_sizes
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return MNISTMLPModel(self._hidden_sizes)

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        key = jax.random.PRNGKey(37)
        img = jax.random.normal(key, (4, 28, 28, 1))  # B, H, W, C
        # Channels is 1 as MNIST is in grayscale.
        return img

    # @override
    def _get_forward_method_args(self):
        inp = self._get_input_activations()

        parameters = self._model.init(jax.random.PRNGKey(42), inp)

        return [parameters, inp]


# ----- Fixtures -----


@pytest.fixture
def inference_tester(request) -> MNISTMLPTester:
    return MNISTMLPTester(request.param)


@pytest.fixture
def training_tester(request) -> MNISTMLPTester:
    return MNISTMLPTester(request.param, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.parametrize(
    "inference_tester",
    [
        (512, 512),
        # (128,),
        # (128, 128),
        # (192, 128),
        # (128, 128, 128),
        # (256, 128, 64),
    ],
    indirect=True,
)
def test_mnist_inference(
    inference_tester: MNISTMLPTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_training(
    training_tester: MNISTMLPTester,
):
    training_tester.test()
