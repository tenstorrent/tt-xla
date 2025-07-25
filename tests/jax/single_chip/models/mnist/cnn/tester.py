# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence, Type

import jax
import jax.numpy as jnp
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, RunMode
from infra.utilities import Model
from jaxtyping import PyTree


class MNISTCNNTester(JaxModelTester):
    """Tester for MNIST CNN model."""

    def __init__(
        self,
        model_class: Type[Model],
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_class = model_class
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return self._model_class()

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        # Channels is 1 as MNIST is in grayscale.
        return jnp.ones((4, 28, 28, 1))  # B, H, W, C

    # @override
    def _get_input_parameters(self) -> PyTree:
        # Example of flax.linen convention of first instatiating a model object
        # and then later calling init to generate a set of initial tensors (parameters
        # and maybe some extra state). Parameters are not stored with the models
        # themselves, they are provided together with inputs to the forward method.
        return self._model.init(
            jax.random.PRNGKey(42), self._input_activations, train=False
        )

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {"train": False}

    # @override
    def _get_static_argnames(self):
        return ["train"]
