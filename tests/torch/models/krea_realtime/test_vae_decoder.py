# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Krea Realtime — AutoencoderKLWan (3D causal VAE) decoder component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, RunMode, run_graph_test
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.krea_realtime_video.pytorch import (
    ModelLoader,
    ModelVariant,
)


# VAE decoder hangs (240-min job timeout, no result); skipped so it can't hang the
# whole job. skip (not xfail) because xfail only applies after the test runs, which
# cannot prevent a hang. See https://github.com/tenstorrent/tt-xla/issues/5678
@pytest.mark.skip(
    reason="VAE decoder hangs (240-min job timeout). https://github.com/tenstorrent/tt-xla/issues/5678"
)
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="KreaRealtimeVideo_VAEDecoder",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_vae_decoder():
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
