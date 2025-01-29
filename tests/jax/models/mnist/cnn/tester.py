# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from infra import ModelTester


class MNISTCNNTester(ModelTester):
    """Tester for MNIST CNN model."""

    def __init__(self, cls):
        self._model_class = cls
        super().__init__()

    # @override
    def _get_model(self) -> nn.Module:
        return self._model_class()

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
