# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
    failed_runtime,
)
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.xlm_roberta.causal_lm.jax import (
    ModelVariant,
    ModelLoader,
)
from ..tester import XLMRobertaTester

MODEL_PATH = "FacebookAI/xlm-roberta-base"
VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> XLMRobertaTester:
    return XLMRobertaTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> XLMRobertaTester:
    return XLMRobertaTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Statically allocated circular buffers on core range [(x=0,y=0) - (x=12,y=9)] "
        "grow to 2101768 B which is beyond max L1 size of 1572864 B "
        "https://github.com/tenstorrent/tt-xla/issues/1066"
    )
)
def test_xlm_roberta_base_inference(inference_tester: XLMRobertaTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_xlm_roberta_base_training(training_tester: XLMRobertaTester):
    training_tester.test()
