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
        model = self._model_class()
        model.params = model.init(jax.random.PRNGKey(42), self._get_input_activations(), train=False)

        return model

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        # Channels is 1 as MNIST is in grayscale.
        return jnp.ones((4, 28, 28, 1))  # B, H, W, C

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {"train": False}

    # @override
    def _get_static_argnames(self):
        return ["train"]
