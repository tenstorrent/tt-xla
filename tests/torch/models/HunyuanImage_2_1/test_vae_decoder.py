# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HunyuanImage 2.1 (Distilled) — AutoencoderKLHunyuanImage decoder component test.

Decodes a denoised latent back to pixel space (32x spatial compression).

IN:  z (1, 64, 64, 64) float
OUT: sample (1, 3, 2048, 2048) float

weight_fit: single_device (0.41B, fits n150 + p150). PCC 0.99.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.hunyuan_image_2_1.pytorch import (
    ModelLoader,
    ModelVariant,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)
    compiler_config = CompilerConfig(optimization_level=1)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
