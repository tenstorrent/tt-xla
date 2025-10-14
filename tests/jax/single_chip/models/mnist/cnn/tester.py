# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, Model, RunMode
from infra.workloads.workload import Workload
from jaxtyping import PyTree

from third_party.tt_forge_models.mnist.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)


class MNISTCNNTester(JaxModelTester):
    """Tester for MNIST CNN model."""

    def __init__(
        self,
        variant: ModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant)
        self._is_dropout = variant == ModelVariant.CNN_DROPOUT

        super().__init__(comparison_config, run_mode, has_batch_norm=True)

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
        kwargs = super()._get_forward_method_kwargs()
        if self._is_dropout and self._run_mode == RunMode.TRAINING:
            kwargs["rngs"] = {"dropout": jax.random.key(1)}
        return kwargs
