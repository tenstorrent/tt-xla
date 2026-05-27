# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CogVideoX-5b — AutoencoderKLCogVideoX decoder component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.cog_videox.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=-0.7399767808716264 - https://github.com/tenstorrent/tt-xla/issues/4791"
)
def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
