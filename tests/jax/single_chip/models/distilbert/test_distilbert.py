# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
import pytest
from infra import ComparisonConfig, JaxModelTester, Model, RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_ttmlir_compilation

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.distilbert.masked_lm.jax import (
    ModelLoader,
    ModelVariant,
)

VARIANT_NAME = ModelVariant.BASE_UNCASED

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Tester -----


class FlaxDistilBertForMaskedLMTester(JaxModelTester):
    """Tester for DistilBert model on a masked language modeling task"""

    def __init__(
        self,
        variant: ModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ):
        self._model_loader = ModelLoader(variant)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        return self._model_loader.load_inputs()

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        kwargs = super()._get_forward_method_kwargs()

        if self._run_mode == RunMode.TRAINING:
            kwargs["dropout_rng"] = jax.random.key(1)
        return kwargs


# ----- Fixtures -----


@pytest.fixture
def training_tester() -> FlaxDistilBertForMaskedLMTester:
    return FlaxDistilBertForMaskedLMTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.test_forge_models_training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'ttir.scatter' "
        "https://github.com/tenstorrent/tt-mlir/issues/4792"
    )
)
def test_flax_distilbert_training(training_tester: FlaxDistilBertForMaskedLMTester):
    training_tester.test()
