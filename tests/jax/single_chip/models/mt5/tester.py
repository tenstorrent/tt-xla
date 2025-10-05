# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from jaxtyping import PyTree

from infra import ComparisonConfig, JaxModelTester, Model, RunMode

from third_party.tt_forge_models.mt5.nlp_summarization.jax import (
    ModelLoader,
    ModelVariant,
)


class MT5Tester(JaxModelTester):
    """Tester for mT5 models."""

    def __init__(
        self,
        variant_name: ModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model(dtype_override=jax.numpy.bfloat16)

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        return self._model_loader.load_inputs(dtype_override=jax.numpy.bfloat16)

    # @overridde
    def _get_forward_method_kwargs(self) -> Dict[str, PyTree]:
        return {
            "params": self._input_parameters,
            **self._input_activations,
        }

    # @override
    def _get_static_argnames(self):
        return ["train"]
