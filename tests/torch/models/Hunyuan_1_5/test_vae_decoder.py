# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo 1.5 — AutoencoderKLHunyuanVideo15 decoder component test."""

import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from third_party.tt_forge_models.hunyuan_1_5.pytorch import ModelLoader, ModelVariant
from loguru import logger

@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            ModelVariant.VAE,
            marks=pytest.mark.xfail(
                reason="hang on TT — https://github.com/tenstorrent/tt-xla/issues/4485",
                strict=False,
            ),
            id="non_tiled",
        ),
        pytest.param(
            ModelVariant.VAE_TILED,
            # marks=pytest.mark.xfail(
            #     reason="ttir.scaled_dot_product_attention rejects 3D mask — https://github.com/tenstorrent/tt-mlir/issues/8362",
            #     strict=False,
            # ),
            id="tiled",
        ),
    ],
)
def test_vae_decoder(variant):
    # Not using run_graph_test: it runs the model on CPU as a golden reference
    # first, and the VAE decoder CPU pass takes too long at full resolution.
    # TT-only for now.
    # TODO: re-enable CPU golden + PCC check via run_graph_test once the TT
    # path is passing.
    xr.set_device_type("TT")
    torch.manual_seed(42)

    device = xm.xla_device()

    loader = ModelLoader(variant)
    model = loader.load_model(dtype_override=torch.bfloat16).to(device)
    logger.info("model={}",model)
    
    compiled = torch.compile(model, backend="tt")

    [z] = loader.load_inputs(dtype_override=torch.bfloat16)
    z = z.to(device)

    with torch.no_grad():
        tt_out = compiled(z)
