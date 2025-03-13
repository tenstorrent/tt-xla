# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import pytest
import torch
from flax import linen as nn
from huggingface_hub import hf_hub_download
from infra import Framework, ModelTester, RunMode
from transformers import AutoTokenizer

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

from .model_implementation import SqueezeBertConfig, SqueezeBertForMaskedLM

MODEL_PATH = "squeezebert/squeezebert-uncased"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "squeezebert",
    "uncased",
    ModelTask.NLP_MASKED_LM,
    ModelSource.CUSTOM,
)

# ----- Tester -----


class SqueezeBertTester(ModelTester):
    """Tester for SqueezeBERT model on a masked language modeling task"""

    # @override
    def _get_model(self) -> nn.Module:
        config = SqueezeBertConfig.from_pretrained(MODEL_PATH)
        return SqueezeBertForMaskedLM(config)

    # @override
    def _get_forward_method_name(self):
        return "apply"

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        inputs = tokenizer("The [MASK] barked at me", return_tensors="np")
        return inputs["input_ids"]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        model_file = hf_hub_download(
            repo_id="squeezebert/squeezebert-uncased", filename="pytorch_model.bin"
        )
        state_dict = torch.load(model_file, weights_only=True)

        params = self._model.init_from_pytorch_statedict(state_dict, dtype=jnp.bfloat16)

        return {
            "variables": params,  # JAX frameworks have a convention of passing weights as the first argument
            "input_ids": self._get_input_activations(),
            "train": False,
        }

    # @override
    def _get_static_argnames(self):
        return ["train"]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> SqueezeBertTester:
    return SqueezeBertTester()


@pytest.fixture
def training_tester() -> SqueezeBertTester:
    return SqueezeBertTester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation("failed to legalize operation 'ttir.gather'")
)
def test_squeezebert_inference(inference_tester: SqueezeBertTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_squeezebert_training(training_tester: SqueezeBertTester):
    training_tester.test()
