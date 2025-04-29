# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from infra.multichip_utils import enable_shardy

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_runtime,
    failed_ttmlir_compilation,
)

from .tester import AlexNetMultichipTester

MODEL_NAME = build_model_name(
    Framework.JAX,
    "alexnet",
    "multichip_n300",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlexNetMultichipTester:
    return AlexNetMultichipTester(run_mode=RunMode.INFERENCE)


@pytest.fixture
def training_tester() -> AlexNetMultichipTester:
    return AlexNetMultichipTester(run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "tt::runtime::ttnn::operations::creation::run: subMesh.num_devices() == 1 "
        "(https://github.com/tenstorrent/tt-xla/issues/548)"
    )
)
def test_alexnet_multichip_n300_inference(inference_tester: AlexNetMultichipTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "'ttir.pooling' op expected type of operand #1 ('tensor<2x26x26x64xbf16>') to match type of corresponding result ('tensor<2x26x26x64xbf16, #tt.mesh_sharding<\"mesh\">>') "
        "(https://github.com/tenstorrent/tt-xla/issues/549)"
    )
)
def test_alexnet_multichip_n300_inference_shardy(
    inference_tester: AlexNetMultichipTester,
):
    with enable_shardy(True):
        inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_alexnet_multichip_n300_training(training_tester: AlexNetMultichipTester):
    training_tester.test()
