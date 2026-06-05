# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Krea Realtime — AutoencoderKLWan (3D causal VAE) decoder component test (~0.13B).

single_device: the VAE fits comfortably on a single n150/p150 chip, so this
runs a real CPU golden + PCC 0.99 check via run_graph_test.

Captured I/O (480x832, 3 latent frames):
  z   [1, 16, 3, 60, 104]  bf16
OUT: [1, 3, 9, 480, 832]   bf16   (causal VAE expands 3 latent -> 9 video frames)

Known risk: a temporal slice on a size-1 dim has failed on TT
(https://github.com/tenstorrent/tt-xla/issues/4465); model-bringup-run/-diagnose
classifies the actual outcome and config-update records bringup_status.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.krea_realtime_video.pytorch import (
    ModelLoader,
    ModelVariant,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # opt_level=1 keeps ttir.group_norm -> ttnn.group_norm (opt_level=0
    # decomposes GroupNorm into reshape+mean+sub, a common DRAM-OOM trigger
    # for diffusion VAEs).
    compiler_config = CompilerConfig(optimization_level=1)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
