# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
import pytest
from utils import (
    BringupStatus,
    Category,
)
from third_party.tt_forge_models.config import Parallelism

from infra import Framework, JaxModelTester, RunMode, Model, ComparisonConfig
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


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxDistilBertForMaskedLMTester:
    return FlaxDistilBertForMaskedLMTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> FlaxDistilBertForMaskedLMTester:
    return FlaxDistilBertForMaskedLMTester(VARIANT_NAME, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.PASSED,
)
def test_flax_distilbert_inference(inference_tester: FlaxDistilBertForMaskedLMTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_distilbert_training(training_tester: FlaxDistilBertForMaskedLMTester):
    training_tester.test()
