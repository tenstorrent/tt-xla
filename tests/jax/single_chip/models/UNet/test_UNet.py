# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn

from infra import ComparisonConfig, Framework, ModelTester, RunMode
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._hidden_channels = hidden_channels
        self._num_levels = num_levels
        super().__init__(comparison_config, run_mode)

    def _get_model(self):
        model = UNet(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            hidden_channels=self._hidden_channels,
            num_levels=self._num_levels,
        )
        return model

    def _get_forward_method_name(self):
        return "apply"

    def _get_input_activations(self):
        key = jax.random.PRNGKey(123)
        return jax.random.normal(key, (1, 572, 572, 1))  # B, H, W, C

    def _get_forward_method_args(self):
        inputs = self._get_input_activations()
        parameters = self._model.init(jax.random.PRNGKey(0), inputs)
        return [parameters, inputs]


# ----- Fixtures -----


@pytest.fixture
def inference_tester(request) -> UNetTester:
    return UNetTester(*request.param)


@pytest.fixture
def training_tester(request) -> UNetTester:
    return UNetTester(*request.param, RunMode.TRAINING)


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
@pytest.mark.parametrize(
    "inference_tester",
    [
        (1, 2, 64, 4),
        (1, 2, 128, 5),
    ],
    indirect=True,
    ids=lambda val: f"in={val[0]}_out={val[1]}_h={val[2]}_lvl={val[3]}",
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
