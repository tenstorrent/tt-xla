# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — AutoencoderKLFlux2 (vae) decoder component test (~0.084B).

single_device: the VAE fits on a single n150/p150 chip, so this runs a real
CPU golden + PCC 0.99 check via run_graph_test.

Captured I/O (64x64): latent [1, 32, 8, 8] bf16 -> sample [1, 3, 64, 64] bf16.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, RunMode, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from utils import BringupStatus, Category

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.flux_2_dev.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=ModelLoader.get_model_info(ModelVariant.FLUX2_DEV_VAE),
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.FLUX2_DEV_VAE)
    # fp32 matches the VAE's stored dtype and keeps the CPU golden numerically
    # comparable for the PCC check.
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

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
