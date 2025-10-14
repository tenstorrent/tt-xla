# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Sequence

import jax
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, Model, RunMode
from jaxtyping import PyTree

from third_party.tt_forge_models.mnist.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

MNIST_MLP_PARAMS_INIT_SEED = 42


def create_mnist_random_input_image() -> jax.Array:
    key = jax.random.PRNGKey(37)
    # B, H, W, C
    # Channels is 1 as MNIST is in grayscale.
    img = jax.random.normal(key, (32, 28, 28, 1))
    return img


class MNISTMLPTester(JaxModelTester):
    """Tester for MNIST MLP model."""

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(ModelVariant.MLP_CUSTOM, hidden_sizes)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        return self._model_loader.load_inputs()

    # @override
    def _get_input_parameters(self) -> PyTree:
        return self._model_loader.load_parameters()

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {}

    # @override
    def _get_static_argnames(self) -> Optional[Sequence[str]]:
        return []
