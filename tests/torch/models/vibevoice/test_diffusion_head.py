# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-1.5B — diffusion prediction head component test. Params: ~123 M.

One denoise step of the acoustic diffusion head, conditioned on the LM hidden
state. Batch 2 because the CFG path concatenates the conditional and negative
branches into a single forward.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.vibevoice.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_diffusion_head():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.DIFFUSION_HEAD)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(model, inputs, framework=Framework.TORCH)
