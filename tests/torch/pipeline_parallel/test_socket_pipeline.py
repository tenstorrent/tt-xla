# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""2-stage pipeline with device-to-device socket activation transfer.

The stage-0 -> stage-1 boundary crosses over a TT-Metal MeshSocket path.
"""

import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_plugin_tt.experimental import pipeline
from utils import Category

LAYER_DIMS = [256, 512, 128]  # 2 Linear layers -> 2 stages
DTYPE = torch.bfloat16
BATCH = 32


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float() - a.flatten().float().mean()
    b = b.flatten().float() - b.flatten().float().mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def _build_stages(seed: int = 0) -> tuple[list[nn.Module], nn.Module]:
    torch.manual_seed(seed)
    l0 = nn.Sequential(nn.Linear(LAYER_DIMS[0], LAYER_DIMS[1], dtype=DTYPE), nn.ReLU())
    l1 = nn.Linear(LAYER_DIMS[1], LAYER_DIMS[2], dtype=DTYPE)
    golden = nn.Sequential(l0, l1)
    return [l0, l1], golden


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.record_test_properties(category=Category.OTHER)
def test_two_stage_socket_pipeline():
    if xr.global_runtime_device_count() < 2:
        pytest.skip("needs >= 2 devices")

    stages, golden_model = _build_stages()
    x = torch.randn(BATCH, LAYER_DIMS[0], dtype=DTYPE)
    golden = golden_model(x)  # CPU golden BEFORE moving layers to device

    model = pipeline(stages)
    assert model.stage_param_devices() == ["xla:0", "xla:1"]

    out = model(x)
    xm.mark_step()

    assert out.shape == (BATCH, LAYER_DIMS[-1])
    measured = _pcc(out, golden)
    assert measured >= 0.99, f"PCC too low: {measured:.5f}"
