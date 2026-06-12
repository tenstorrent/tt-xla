# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CogVideoX-5b — T5 v1.1-XXL text encoder component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.cog_videox.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9685751315163685 - https://github.com/tenstorrent/tt-xla/issues/4647"
)
def test_text_encoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
