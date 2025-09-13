# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, RunMode, Model
from jaxtyping import PyTree
from third_party.tt_forge_models.mnist.image_classification.jax import (
    ModelLoader,
    ModelArchitecture,
)


class MNISTMLPTester(JaxModelTester):
    """Tester for MNIST MLP model."""

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(ModelArchitecture.MLP, hidden_sizes)
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
