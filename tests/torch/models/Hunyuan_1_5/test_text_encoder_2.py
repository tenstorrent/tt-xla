# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo 1.5 — ByT5 text encoder component test (text_encoder_2)."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.hunyuan_1_5.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="PCC drop (got 0.9712937232386395, required >= 0.99) — https://github.com/tenstorrent/tt-xla/issues/4484"
)
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_text_encoder_2():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER_2)
    encoder = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        encoder,
        inputs,
        framework=Framework.TORCH,
    )
