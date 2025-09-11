# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
<<<<<<< HEAD
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
)
from third_party.tt_forge_models.gpt2.causal_lm.jax import ModelVariant
from ..tester import GPT2Tester

MODEL_VARIANT = ModelVariant.BASE
MODEL_NAME = build_model_name(
    Framework.JAX,
    "gpt2",
    "base",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)
=======
)
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.gpt2.causal_lm.jax import (
    ModelVariant,
    ModelLoader,
)
from ..tester import GPT2Tester

VARIANT_NAME = ModelVariant.BASE

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)
>>>>>>> bbae47f6 (refactored models to use modelinfo set-1)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPT2Tester:
<<<<<<< HEAD
    return GPT2Tester(MODEL_VARIANT)
=======
    return GPT2Tester(VARIANT_NAME)
>>>>>>> bbae47f6 (refactored models to use modelinfo set-1)


@pytest.fixture
def training_tester() -> GPT2Tester:
<<<<<<< HEAD
    return GPT2Tester(MODEL_VARIANT, run_mode=RunMode.TRAINING)
=======
    return GPT2Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)
>>>>>>> bbae47f6 (refactored models to use modelinfo set-1)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.PASSED,
)
def test_gpt2_base_inference(inference_tester: GPT2Tester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt2_base_training(training_tester: GPT2Tester):
    training_tester.test()
