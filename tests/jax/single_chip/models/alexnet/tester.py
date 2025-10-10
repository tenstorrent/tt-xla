# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, RunMode
from jaxtyping import PyTree

from .model_implementation import AlexNetModel

ALEXNET_PARAMS_INIT_SEED = 42


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
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return AlexNetModel()

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        return create_alexnet_random_input_image()

    # @override
    def _get_input_parameters(self) -> PyTree:
        # Example of flax.linen convention of first instatiating a model object
        # and then later calling init to generate a set of initial tensors (parameters
        # and maybe some extra state). Parameters are not stored with the models
        # themselves, they are provided together with inputs to the forward method.
        return self._model.init(
            jax.random.PRNGKey(ALEXNET_PARAMS_INIT_SEED),
            self._input_activations,
            train=False if self._run_mode == RunMode.INFERENCE else True,
        )

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        kwargs = {"train": False if self._run_mode == RunMode.INFERENCE else True}
        if self._run_mode == RunMode.TRAINING:
            kwargs["rngs"] = {"dropout": jax.random.key(1)}
        return kwargs

    # @override
    def _get_static_argnames(self):
        return ["train"]
