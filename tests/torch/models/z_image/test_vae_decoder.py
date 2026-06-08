# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Z-Image — AutoencoderKL decoder component test."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.z_image.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.z_image.pytorch.src.model_utils import SEED

@pytest.mark.model_test
# @pytest.mark.skip(
#     reason=(
#         "TT DRAM OOM at pipeline 1280x720 — "
#         "https://github.com/tenstorrent/tt-xla/issues/4755"
#     )
# )
def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
