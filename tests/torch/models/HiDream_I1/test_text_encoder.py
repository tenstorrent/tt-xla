# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HiDream-I1-Fast — CLIP-L (text_encoder) component test. Params: ~0.123 B."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.hidream_i1.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.testing
def test_text_encoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
