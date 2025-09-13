# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode, Model
from jaxtyping import PyTree
from third_party.tt_forge_models.mistral.causal_lm.jax import ModelLoader, ModelVariant


class Mistral7BTester(JaxModelTester):
    """Tester for Mistral-7B model variants with a language modeling head on top."""

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


class Mistral7BV02Tester(Mistral7BTester):
    """Tester for Mistral-7B model v0.2 and later variants, with a language modeling
    head on top."""

    # The v0.2+ sliding window logic is now handled in the ModelLoader
    pass
