# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode

from ..tester import FlaxRobertaForMaskedLMTester

MODEL_PATH = "FacebookAI/roberta-base"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxRobertaForMaskedLMTester:
    return FlaxRobertaForMaskedLMTester()


@pytest.fixture
def training_tester() -> FlaxRobertaForMaskedLMTester:
    return FlaxRobertaForMaskedLMTester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(reason="failed to legalize operation 'stablehlo.reduce_window'")
def test_flax_roberta_base_inference(
    inference_tester: FlaxRobertaForMaskedLMTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_roberta_base_training(
    training_tester: FlaxRobertaForMaskedLMTester,
):
    training_tester.test()
