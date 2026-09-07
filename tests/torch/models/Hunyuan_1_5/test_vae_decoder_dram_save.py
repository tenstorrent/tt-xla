# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo 1.5 — AutoencoderKLHunyuanVideo15 decoder component test."""

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.infra.testers.compiler_config import CompilerConfig
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
        # Decoder pixel-shuffle permutes end in a dim of size 2, which pads 16x
        # in TILE layout (up_blocks[2]: 0.82 GB -> 13.95 GB DRAM). This pass
        # (tt-mlir #7729 / #8568) runs those permutes in row-major instead.
        compiler_config=CompilerConfig(
            experimental_enable_dram_space_saving_optimization=True
        ),
    )