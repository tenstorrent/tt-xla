# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Sequence

import jax
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, RunMode
from jaxtyping import PyTree

from third_party.tt_forge_models.alexnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)


def create_alexnet_random_input_image() -> jax.Array:
    prng_key = jax.random.PRNGKey(23)
    img = jax.random.randint(
        key=prng_key,
        # B, H, W, C
        shape=(8, 224, 224, 3),
        # In the original paper inputs are normalized with individual channel
        # values learned from training set.
        minval=-128,
        maxval=128,
    )
    return img


class AlexNetTester(JaxModelTester):
    """Tester for AlexNet CNN model."""

    def __init__(
        self,
        variant: ModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return self._model_loader.load_model()

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        return self._model_loader.load_inputs()

    # @override
    def _get_input_parameters(self) -> PyTree:
        return self._model_loader.load_parameters(
            train=self._run_mode == RunMode.TRAINING
        )

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        kwargs = {"train": False if self._run_mode == RunMode.INFERENCE else True}
        if self._run_mode == RunMode.TRAINING:
            kwargs["rngs"] = {"dropout": jax.random.key(1)}
        return kwargs
