# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — FluxTransformer2DModel component test (1024x1024 latent geometry)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.model_test
def test_transformer():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        # DRAM space-saving frees the sliver to fit ~23.8GB bf16 weights single-chip (issue #5251).
        compiler_config=CompilerConfig(
            experimental_enable_dram_space_saving_optimization=True
        ),
    )
