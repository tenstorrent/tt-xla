# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from op_by_op_infra.pydantic_models import model_to_dict
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
)

from .tester import MNISTMLPTester

MODEL_NAME = build_model_name(
    Framework.JAX,
    "mnist",
    "mlp",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester(request) -> MNISTMLPTester:
    return MNISTMLPTester(request.param)


@pytest.fixture
def inference_op_by_op_tester(request) -> MNISTMLPTester:
    return MNISTMLPTester(request.param, run_mode=RunMode.INFERENCE_OP_BY_OP)


@pytest.fixture
def training_tester(request) -> MNISTMLPTester:
    return MNISTMLPTester(request.param, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
)
@pytest.mark.parametrize(
    "inference_tester",
    [
        (128,),
        (128, 128),
        (192, 128),
        (512, 512),
        (128, 128, 128),
    ],
    indirect=True,
    ids=lambda val: f"{val}",
)
def test_mnist_mlp_inference_nightly(inference_tester: MNISTMLPTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
@pytest.mark.parametrize(
    "inference_tester", [(256, 128, 64)], indirect=True, ids=lambda val: f"{val}"
)
def test_mnist_mlp_inference(inference_tester: MNISTMLPTester):
    inference_tester.test()


@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE_OP_BY_OP,
)
@pytest.mark.parametrize(
    "inference_op_by_op_tester",
    [(256, 128, 64)],
    indirect=True,
    ids=lambda val: f"{val}",
)
def test_mnist_mlp_inference_op_by_op(
    inference_op_by_op_tester: MNISTMLPTester, record_property
):
    results = inference_op_by_op_tester.test_op_by_op(
        frontend="tt-xla",
        model_name=MODEL_NAME,
    )

    for result in results:
        # TODO dump in format which is suitable for parser to parse.
        record_property(f"OpTest model for: {result.op_name}", model_to_dict(result))


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_mlp_training(training_tester: MNISTMLPTester):
    training_tester.test()
