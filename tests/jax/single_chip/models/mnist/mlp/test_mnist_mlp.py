# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
)

from tests.infra.utilities.utils import create_jax_inference_tester

from .tester import MNISTMLPTester

MODEL_NAME = build_model_name(
    Framework.JAX,
    "mnist",
    "mlp",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


def create_inference_tester(hidden_sizes: tuple, format: str) -> MNISTMLPTester:
    """Create inference tester with specified hidden sizes and data format."""
    return create_jax_inference_tester(MNISTMLPTester, hidden_sizes, format)


@pytest.fixture
def inference_tester(request) -> MNISTMLPTester:
    return MNISTMLPTester(request.param)


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
    bringup_status=BringupStatus.PASSED,
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
@pytest.mark.parametrize("hidden_sizes", [(256, 128, 64)], ids=lambda val: f"{val}")
@pytest.mark.parametrize(
    "format",
    [
        "float32",
        "bfloat16",
        pytest.param(
            "bfp8",
            marks=pytest.mark.skip(
                reason="Skip until mixed-precision is supported in MLIR. https://github.com/tenstorrent/tt-mlir/issues/5252"
            ),
        ),
    ],
)
def test_mnist_mlp_inference(hidden_sizes: tuple, format: str, request):
    tester = create_inference_tester(hidden_sizes, format)
    tester.test()

    if request.config.getoption("--serialize", default=False):
        inference_tester.serialize_compilation_artifacts(request.node.name)


@pytest.mark.push
@pytest.mark.training
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.PASSED,
)
@pytest.mark.parametrize(
    "training_tester", [(256, 128, 64)], indirect=True, ids=lambda val: f"{val}"
)
def test_mnist_mlp_training(training_tester: MNISTMLPTester):
    training_tester.test()
