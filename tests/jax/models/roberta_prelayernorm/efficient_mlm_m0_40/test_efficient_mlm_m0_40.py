# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Sequence

import jax
import pytest
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import (
    AutoTokenizer,
    FlaxPreTrainedModel,
    FlaxRobertaPreLayerNormForMaskedLM,
)
from utils import record_model_test_properties, runtime_fail

MODEL_PATH = "andreasmadsen/efficient_mlm_m0.40"
MODEL_NAME = "roberta-prelayernorm"


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


@pytest.mark.nightly
@pytest.mark.xfail(
    reason=runtime_fail(
        "Unsupported data type DataType::INT32 "
        "(https://github.com/tenstorrent/tt-xla/issues/308)"
    )
)
def test_flax_roberta_prelayernorm_inference(
    inference_tester: FlaxRobertaPreLayerNormForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_roberta_prelayernorm_training(
    training_tester: FlaxRobertaPreLayerNormForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
