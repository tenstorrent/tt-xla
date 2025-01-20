# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode

from ..tester import LLamaTester

MODEL_PATH = "openlm-research/open_llama_3b_v2"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> LLamaTester:
    return LLamaTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> LLamaTester:
    return LLamaTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(reason="failed to legalize operation 'stablehlo.reduce'")
def test_openllama3b_inference(
    inference_tester: LLamaTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_openllama3b_training(
    training_tester: LLamaTester,
):
    training_tester.test()
