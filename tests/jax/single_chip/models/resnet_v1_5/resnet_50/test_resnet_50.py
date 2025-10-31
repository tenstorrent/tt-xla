# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from infra.comparators import ComparisonConfig, PccConfig
from pytest import MonkeyPatch
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    ModelGroup,
    failed_ttmlir_compilation,
    incorrect_result,
)

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.resnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import CompilerConfig, ResNetTester

VARIANT_NAME = ModelVariant.RESNET_50
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def trace_tester(monkeypatch: MonkeyPatch) -> ResNetTester:
    # These need to be set before the tester is created
    monkeypatch.setenv("TT_RUNTIME_TRACE_REGION_SIZE", "10000000")

    cc = CompilerConfig(enable_optimizer=True, enable_trace=True)
    return ResNetTester(VARIANT_NAME, compiler_config=cc)


@pytest.fixture
def training_tester() -> ResNetTester:
    return ResNetTester(
        VARIANT_NAME,
        run_mode=RunMode.TRAINING,
    )


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    model_group=ModelGroup.RED,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
@pytest.mark.large
def test_resnet_v1_5_50_inference_trace(
    trace_tester: ResNetTester,
):
    trace_tester.test()


@pytest.mark.push
@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'stablehlo.pad' "
        "https://github.com/tenstorrent/tt-mlir/issues/5305"
    )
)
def test_resnet_v1_5_50_training(training_tester: ResNetTester):
    training_tester.test()
