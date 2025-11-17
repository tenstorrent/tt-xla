# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
import pytest
from flax import nnx
from infra import JaxModelTester, RunMode
from jaxtyping import PyTree

from .tester import GPT2EasyDel


@pytest.fixture
def inference_tester() -> GPT2EasyDel:
    return GPT2EasyDel(axis_names=("x", "y"), mesh_shape=(1, 1), run_mode=RunMode.INFERENCE)


@pytest.fixture
def training_tester() -> GPT2EasyDel:
    return GPT2EasyDel(axis_names=("x", "y"), mesh_shape=(1, 1), run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
def test_simple_mlp_inference(inference_tester: GPT2EasyDel):
    inference_tester.test()

@pytest.mark.skip(
    reason=("Easydel training not supported yet")
)
@pytest.mark.push
def test_simple_mlp_training(training_tester: GPT2EasyDel):
    training_tester.test()
