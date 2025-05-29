# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    incorrect_result,
)

from ..tester import AlbertV2Tester

info = ModelInfo(
    "albert_v2",
    "base",
    "albert/albert-base-v2",
    ModelGroup.GENERALITY,
    ModelTask.NLP_MASKED_LM,
    ModelSource.HUGGING_FACE,
    Framework.JAX,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlbertV2Tester:
    return AlbertV2Tester(info.path)


@pytest.fixture
def training_tester() -> AlbertV2Tester:
    return AlbertV2Tester(info.path, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
    model_info=info,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "Atol comparison failed. Calculated: atol=34.321659088134766. Required: atol=0.16 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_flax_albert_v2_base_inference(inference_tester: AlbertV2Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    run_mode=RunMode.TRAINING,
    model_info=info,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_albert_v2_base_training(training_tester: AlbertV2Tester):
    training_tester.test()
