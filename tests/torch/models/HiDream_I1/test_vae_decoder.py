# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HiDream-I1-Fast — AutoencoderKL (vae) decoder component test. Params: ~0.084 B."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.hidream_i1.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason=" Out of Memory: Not enough space to allocate 4294967296 B DRAM buffer across 12 banks, where each bank needs to store 357916672 B, but bank size is 1071821792 B - https://github.com/tenstorrent/tt-xla/issues/4762"
)
@pytest.mark.nightly
@pytest.mark.model_test
def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
