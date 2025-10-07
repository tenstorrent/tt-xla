# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from tests.infra.comparators.comparison_config import ComparisonConfig, PccConfig
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    failed_ttmlir_compilation,
)

from ..tester import RoFormerTester
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.roformer.masked_lm.jax import ModelLoader, ModelVariant

VARIANT_NAME = ModelVariant.CHINESE_BASE
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> RoFormerTester:
    # Reduced PCC threshold - #1454
    return RoFormerTester(
        VARIANT_NAME,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
    )


@pytest.fixture
def training_tester() -> RoFormerTester:
    return RoFormerTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
def test_roformer_chinese_base_inference(inference_tester: RoFormerTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'ttir.scatter' "
        "https://github.com/tenstorrent/tt-mlir/issues/4792"
    )
)
def test_roformer_chinese_base_training(training_tester: RoFormerTester):
    training_tester.test()
