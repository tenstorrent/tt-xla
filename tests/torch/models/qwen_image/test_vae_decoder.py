# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image — AutoencoderKLQwenImage decoder component test."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.qwen_image.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.qwen_image.pytorch.src.model_utils import SEED

from . import skip_on_wormhole


@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.model_test
def test_vae_decoder():
    # optimization_level=1 keeps GroupNorm as the native ttnn.group_norm kernel
    # instead of decomposing it; the decomposed mean/subtract materializes a large
    # intermediate at 1024x1024 and can OOM (cf. Z-Image VAE, issue #4755).
    skip_on_wormhole("VAE decoder")
    torch.manual_seed(SEED)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    compiler_config = CompilerConfig(optimization_level=1)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
