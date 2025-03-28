# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, ModelTester, RunMode
import jax
import jax.numpy as jnp
from .model_implementation import UNet


from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

MODEL_NAME = build_model_name(
    Framework.JAX,
    "unet",
    None,
    ModelTask.CV_IMAGE_SEG,
    ModelSource.CUSTOM,
)


class UNetTester(ModelTester):
    """Tester for UNet model"""

    def _get_model(self):
        model = UNet(in_channels=1, out_channels=2, hidden_channels=64, num_levels=4)
        return model

    def _get_forward_method_name(self):
        return "apply"

    def _get_input_activations(self):
        return jnp.ones((1, 572, 572, 1), dtype=jnp.float32)

    def _get_forward_method_kwargs(self):
        rng = jax.random.PRNGKey(0)
        variables = self._model.init(rng, self._get_input_activations())

        return {
            "variables": variables,
            "x": self._get_input_activations(),
        }

    def _get_static_argnames(self):
        return []


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> UNetTester:
    return UNetTester()


@pytest.fixture
def training_tester() -> UNetTester:
    return UNetTester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "Test failed due to missing decomposition from ttir.convolution to ttir.conv_transpose2d."
        "https://github.com/tenstorrent/tt-xla/issues/417"
    )
)
def test_unet_inference(inference_tester: UNetTester):
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
def test_unet_training(training_tester: UNetTester):
    training_tester.test()
