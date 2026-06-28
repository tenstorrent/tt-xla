# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SDXL-Lightning — UNet2DConditionModel (unet) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.sdxl_lightning.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_unet():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.UNET)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
