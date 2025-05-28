# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
from flax import linen as nn
from infra import ComparisonConfig, ModelTester, RunMode
from jaxtyping import PyTree

from .model_implementation import MNISTMLPModel

MNIST_MLP_PARAMS_INIT_SEED = 42


def create_mnist_random_input_image() -> jax.Array:
    key = jax.random.PRNGKey(37)
    # B, H, W, C
    # Channels is 1 as MNIST is in grayscale.
    img = jax.random.normal(key, (4, 28, 28, 1))
    return img


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
        return create_mnist_random_input_image()

    # @override
    def _get_input_parameters(self) -> PyTree:
        return self._model.init(jax.random.PRNGKey(42), self._input_activations)
