# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import ComparisonConfig, Framework, RunMode
from transformers import (
    FlaxBlenderbotSmallForConditionalGeneration,
    FlaxPreTrainedModel,
)
from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

from ..tester import BlenderBotTester

MODEL_PATH = "facebook/blenderbot_small-90M"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "blenderbot",
    "small-90m",
    ModelTask.NLP_SUMMARIZATION,
    ModelSource.HUGGING_FACE,
)


class BlenderBotSmallTester(BlenderBotTester):
    """Tester for BlenderBot Model small variant."""

    def __init__(
        self,
        model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        super().__init__(model_path, comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        return FlaxBlenderbotSmallForConditionalGeneration.from_pretrained(
            self._model_path
        )


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> BlenderBotSmallTester:
    return BlenderBotSmallTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> BlenderBotSmallTester:
    return BlenderBotSmallTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push  # verify in CI that it works
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
        "'ttir.scatter' op Dimension size to slice into must be 1 "
        "https://github.com/tenstorrent/tt-xla/issues/386 "
    )
)
def test_blenderbot_small_90m_inference(inference_tester: BlenderBotSmallTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_blenderbot_small_90m_training(training_tester: BlenderBotSmallTester):
    training_tester.test()
