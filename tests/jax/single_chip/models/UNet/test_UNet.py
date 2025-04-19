# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn

from infra import ComparisonConfig, Framework, ModelTester, RunMode, random_tensor
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
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._in_channels = 1
        self._out_channels = 2
        self._hidden_channels = 64
        self._num_levels = 4
        super().__init__(comparison_config, run_mode)

    def _get_model(self) -> nn.Module:
        model = UNet(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            hidden_channels=self._hidden_channels,
            num_levels=self._num_levels,
        )
        return model

    def _get_forward_method_name(self) -> str:
        return "apply"

    def _get_input_activations(self) -> jnp.ndarray:
        return random_tensor(
            shape=(1, 572, 572, 1),  # B, H, W, C
            dtype=jnp.float32,
            random_seed=123,
            minval=-1.0,
            maxval=1.0,
            framework=Framework.JAX,
        )

    def _get_forward_method_args(self) -> list:
        inputs = self._get_input_activations()
        parameters = self._model.init(
            jax.random.PRNGKey(0),
            inputs,
            train=False,
        )
        return [parameters, inputs]

    def _get_forward_method_kwargs(self) -> dict:
        train = self._run_mode == RunMode.TRAINING
        return {
            "train": train,
        }

    def _get_static_argnames(self) -> list:
        return ["train"]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> UNetTester:
    return UNetTester()


@pytest.fixture
def training_tester() -> UNetTester:
    return UNetTester(run_mode=RunMode.TRAINING)


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
