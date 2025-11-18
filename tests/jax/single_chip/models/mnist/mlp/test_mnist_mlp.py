# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
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


# ----- Tests -----


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
        tester.serialize_compilation_artifacts(request.node.name)