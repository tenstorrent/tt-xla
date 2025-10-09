# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import jax
import pytest
from flax import linen as nn
from infra import ComparisonConfig, Framework, JaxModelTester, Model, RunMode
from jaxtyping import PyTree
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

from third_party.tt_forge_models.mlp_mixer.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

VARIANT_NAME = ModelVariant.BASE_16
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mlpmixer",
    None,
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


class MlpMixerTester(JaxModelTester):
    """Tester for MlpMixer model."""

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(VARIANT_NAME)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> jax.Array:
        return self._model_loader.load_inputs()

    # @override
    def _get_input_parameters(self) -> PyTree:
        return self._model_loader.load_parameters()


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MlpMixerTester:
    return MlpMixerTester()


@pytest.fixture
def training_tester() -> MlpMixerTester:
    return MlpMixerTester(run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed. Calculated: pcc=-0.006597555708140135. Required: pcc=0.99 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_mlpmixer_inference(inference_tester: MlpMixerTester):
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
def test_mlpmixer_training(training_tester: MlpMixerTester):
    training_tester.test()
