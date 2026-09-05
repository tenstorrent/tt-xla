# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo 1.5 — AutoencoderKLHunyuanVideo15 decoder component test."""

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.hunyuan_1_5.pytorch import ModelLoader, ModelVariant
from loguru import logger


def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.bfloat16)
    logger.info(f"Model : {model}")
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
