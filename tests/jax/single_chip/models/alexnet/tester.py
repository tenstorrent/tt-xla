# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

import jax
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, RunMode, Model
from jaxtyping import PyTree
from third_party.tt_forge_models.alexnet.image_classification.jax import ModelLoader, ModelVariant

ALEXNET_PARAMS_INIT_SEED = 42


class AlexNetTester(JaxModelTester):
    """Tester for AlexNet CNN model."""

    def __init__(
        self,
        variant_name: Optional[ModelVariant] = None,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> jax.Array:
        return self._model_loader.load_inputs()

    # @override
    def _get_input_parameters(self) -> PyTree:
        # Example of flax.linen convention of first instatiating a model object
        # and then later calling init to generate a set of initial tensors (parameters
        # and maybe some extra state). Parameters are not stored with the models
        # themselves, they are provided together with inputs to the forward method.
        return self._model.init(
            jax.random.PRNGKey(ALEXNET_PARAMS_INIT_SEED),
            self._input_activations,
            train=False,
        )

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {"train": False}

    # @override
    def _get_static_argnames(self):
        return ["train"]
