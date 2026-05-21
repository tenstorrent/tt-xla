# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Playground v2.5 — CLIPTextModelWithProjection (text_encoder_2) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.playground_v2_5.pytorch import (
    ModelLoader,
    ModelVariant,
)
from loguru import logger

# @pytest.mark.xfail(
#     reason="AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9711299232660258. Required: pcc=0.99 — https://github.com/tenstorrent/tt-xla/issues/4709"
# )
@pytest.mark.nightly
@pytest.mark.model_test
def test_text_encoder_2():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER_2)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)
    
    logger.info("model={}",model)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
