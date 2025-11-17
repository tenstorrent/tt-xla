# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Sequence

import jax
import pytest
from flax import linen as nn
from infra import ComparisonConfig, Framework, JaxModelTester, Model, RunMode
from jaxtyping import PyTree
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_runtime,
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

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {}

    # @override
    def _get_static_argnames(self) -> Optional[Sequence[str]]:
        return []


# ----- Fixtures -----


@pytest.fixture
def training_tester() -> MlpMixerTester:
    return MlpMixerTester(run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.test_forge_models_training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 36126720 B L1 buffer across 24 banks, "
        "where each bank needs to store 1505280 B, but bank size is only 1331936 B "
        "https://github.com/tenstorrent/tt-xla/issues/918"
    )
)
def test_mlpmixer_training(training_tester: MlpMixerTester):
    training_tester.test()
