# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SDXL-Lightning — AutoencoderKL (vae) decoder component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.sdxl_lightning.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    # opt_level=1 keeps ttir.group_norm -> ttnn.group_norm; opt_level=0 triggers
    # GroupNorm decomposition (reshape+mean+sub) which triggers OOM
    # https://github.com/tenstorrent/tt-xla/issues/4710
    compiler_config = CompilerConfig(optimization_level=1)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
