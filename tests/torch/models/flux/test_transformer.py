# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — FluxTransformer2DModel component test (128x128 latent geometry)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="Single-chip device DRAM OOM during execution. On Blackhole (~34 GB: "
    "8 banks x 4.27 GB) the ~23.8 GB weights fit and execution starts, but an "
    "intermediate ~4.8 GB activation buffer can't allocate (~30 GB already used, "
    "<0.5 GB/bank free) - marginally over; needs memory opt (bfp8 weights / "
    "optimization_level=2). On Wormhole (12 GB) it OOMs outright. The full "
    "run_graph_test also needs >31 GB host RAM (weights + trace + CPU reference). "
    "skip (not xfail) because a host OOM-kill / device TT_FATAL aborts the "
    "process rather than raising. Tracking issue TBD."
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
