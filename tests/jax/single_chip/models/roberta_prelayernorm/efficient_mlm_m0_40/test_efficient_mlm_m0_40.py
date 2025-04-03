# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import pytest
from infra import ComparisonConfig, Framework, ModelTester, RunMode
from transformers import (
    AutoTokenizer,
    FlaxPreTrainedModel,
    FlaxRobertaPreLayerNormForMaskedLM,
)

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_fe_compilation,
    incorrect_result,
)

MODEL_PATH = "andreasmadsen/efficient_mlm_m0.40"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "roberta_prelayernorm",
    "efficient_mlm_m0.40",
    ModelTask.NLP_MASKED_LM,
    ModelSource.HUGGING_FACE,
)


class FlaxRobertaPreLayerNormForMaskedLMTester(ModelTester):
    """Tester for Roberta PreLayerNorm model on a masked language modeling task."""

    def __init__(
        self,
        model_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_name = model_name
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        return FlaxRobertaPreLayerNormForMaskedLM.from_pretrained(
            self._model_name, from_pt=True
        )

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        inputs = tokenizer("Hello <mask>.", return_tensors="np")
        return inputs["input_ids"]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "input_ids": self._get_input_activations(),
        }

    # @ override
    def _get_static_argnames(self):
        return ["train"]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxRobertaPreLayerNormForMaskedLMTester:
    return FlaxRobertaPreLayerNormForMaskedLMTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxRobertaPreLayerNormForMaskedLMTester:
    return FlaxRobertaPreLayerNormForMaskedLMTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


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
        "Atol comparison failed. Calculated: atol=131048.65625. Required: atol=0.16 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_flax_roberta_prelayernorm_inference(
    inference_tester: FlaxRobertaPreLayerNormForMaskedLMTester,
):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_roberta_prelayernorm_training(
    training_tester: FlaxRobertaPreLayerNormForMaskedLMTester,
):
    training_tester.test()
