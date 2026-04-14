# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal reproducer for the OFT model OOM.

The OFT model uses F.grid_sample to sample an integral image at bounding-box
locations derived from a 160x160 bird's-eye-view grid. Each grid_sample call
takes [1, 256, 28, 28] input and [1, 7, 25281, 2] grid, producing a
[1, 256, 7, 25281] output. When lowered to StableHLO this becomes a gather
with a [1, 256, 7, 25281, 4] i64 index tensor requiring ~1.35 GiB DRAM —
exceeding the ~1 GiB bank size.

This causes:
  TT_FATAL: Out of Memory: Not enough space to allocate 1449713664 B DRAM buffer
  across 12 banks

Isolated from: oft/pytorch -> OFT.forward -> F.grid_sample(integral_img, bbox_corners)
"""

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from infra import Framework
from infra.testers.single_chip.graph.graph_tester import run_graph_test

INPUTS_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "oft_grid_sample_inputs.pt"
)


def _get_inputs():
    """Load real inputs captured from the OFT model forward pass.

    integral_img: [1, 256, 28, 28] — cumulative sum of feature map (lat8)
    grid_tl:      [1, 7, 25281, 2] — grid coords clamped to [-1, 1]

    If saved file doesn't exist, generate representative inputs with
    the exact same shapes and value characteristics.
    """
    if os.path.exists(INPUTS_SAVE_PATH):
        data = torch.load(INPUTS_SAVE_PATH, weights_only=True)
        return data["integral_img"], data["grid_tl"]

    # Fallback: generate inputs with correct shapes.
    # integral_img is a cumulative sum so values are non-negative and grow.
    features = torch.randn(1, 256, 28, 28).abs()
    integral_img = torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
    # grid values are normalized bbox corners clamped to [-1, 1]
    grid = torch.randn(1, 7, 25281, 2).clamp(-1, 1)
    torch.save({"integral_img": integral_img, "grid_tl": grid}, INPUTS_SAVE_PATH)
    return integral_img, grid


class OFTSingleGridSample(nn.Module):
    """Single grid_sample with OFT model shapes — the minimal OOM trigger."""

    def forward(self, integral_img, grid):
        return F.grid_sample(integral_img, grid, align_corners=False)


@pytest.mark.single_device
def test_oft_single_grid_sample(request):
    """Single F.grid_sample: [1,256,28,28] x [1,7,25281,2] -> [1,256,7,25281].

    Inputs are captured from the real OFT model forward pass (OFT8 module).
    The gather index tensor [1,256,7,25281,4] in i64 requires ~1.35 GiB,
    exceeding the ~1 GiB DRAM bank size.
    """
    model = OFTSingleGridSample()
    integral_img, grid = _get_inputs()
    run_graph_test(
        model, [integral_img, grid], framework=Framework.TORCH, request=request
    )
