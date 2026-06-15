# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — FluxTransformer2DModel component test (128x128 latent geometry)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="Model OOM on a single chip: TT_FATAL Out of Memory allocating a "
    "~12.99 GB DRAM buffer across 12 banks (bank size ~1.07 GB). The ~12B "
    "FLUX.1-dev transformer exceeds single-chip DRAM at 128x128. Needs the "
    "multi-chip sharded path (later bringup phase). Tracking issue TBD."
)
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
    )
