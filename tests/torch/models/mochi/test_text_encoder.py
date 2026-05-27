# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mochi — T5-XXL text encoder component test (4.76B)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.mochi.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.mochi.pytorch.src.utils import (
    load_text_encoder_inputs_full_res,
)


@pytest.mark.xfail(
    reason="AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9651207601058263. Required: pcc=0.99 - https://github.com/tenstorrent/tt-xla/issues/4652 "
)
def test_text_encoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.MOCHI, subfolder="text_encoder")
    encoder = loader.load_model(dtype_override=torch.bfloat16)
    inputs = load_text_encoder_inputs_full_res(dtype=torch.bfloat16)

    run_graph_test(encoder, inputs, framework=Framework.TORCH)
