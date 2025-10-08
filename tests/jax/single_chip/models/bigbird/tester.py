# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence
import jax
from infra import ComparisonConfig, JaxModelTester, RunMode, Model
from third_party.tt_forge_models.bigbird.causal_lm.jax.loader import (
    ModelLoader as BigBirdCLMModelLoader,
    ModelVariant as BigBirdCLMModelVariant,
)
from third_party.tt_forge_models.bigbird.question_answering.jax.loader import (
    ModelLoader as BigBirdQAModelLoader,
    ModelVariant as BigBirdQAModelVariant,
)


class BigBirdQATester(JaxModelTester):
    """Tester for BigBird Question Answering Model variants."""

    def __init__(
        self,
        variant_name: BigBirdQAModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = BigBirdQAModelLoader(variant_name)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        return self._model_loader.load_inputs()

    # @override
    def _wrapper_model(self, f, is_hf_model=True):
        def model(args, kwargs):
            out = f(*args, **kwargs)
            out = out.end_logits
            return out

        return model


class BigBirdCLMTester(JaxModelTester):
    """Tester for BigBird Causal LM Model variants."""

    def __init__(
        self,
        variant_name: BigBirdCLMModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = BigBirdCLMModelLoader(variant_name)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        return self._model_loader.load_inputs()
