# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode

from ..tester import FlaxRobertaForMaskedLMTester

MODEL_PATH = "FacebookAI/roberta-large"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxRobertaForMaskedLMTester:
    return FlaxRobertaForMaskedLMTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxRobertaForMaskedLMTester:
    return FlaxRobertaForMaskedLMTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(
    reason="Cannot get the device from a tensor with host storage (https://github.com/tenstorrent/tt-xla/issues/171)"
)
def test_flax_roberta_large_inference(
    inference_tester: FlaxRobertaForMaskedLMTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_roberta_large_training(
    training_tester: FlaxRobertaForMaskedLMTester,
):
    training_tester.test()
