# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — AutoencoderKLFlux2 decoder component test (1024x1024)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.flux2.pytorch import ModelLoader, ModelVariant

from . import skip_on_wormhole


# VAE decoder hangs (240-min job timeout, no result); skipped so it can't hang the
# whole job. skip (not xfail) because xfail only applies after the test runs, which
# cannot prevent a hang. See https://github.com/tenstorrent/tt-xla/issues/5678
@pytest.mark.skip(
    reason="VAE decoder hangs (240-min job timeout). https://github.com/tenstorrent/tt-xla/issues/5678"
)
@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.model_test
def test_vae_decoder():
    skip_on_wormhole("VAE decoder")
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
