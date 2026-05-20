# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo 1.5 — AutoencoderKLHunyuanVideo15 decoder component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.hunyuan_1_5.pytorch import ModelLoader, ModelVariant


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
            marks=pytest.mark.xfail(
                reason="ttir.scaled_dot_product_attention rejects 3D mask — https://github.com/tenstorrent/tt-mlir/issues/8362",
                strict=False,
            ),
            id="tiled",
        ),
    ],
)
def test_vae_decoder(variant):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
