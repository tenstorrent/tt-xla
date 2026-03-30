# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test reproducing the Conv2d DRAM slicing failure from
bevdepth/pytorch-bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da.

Failing op:  ttnn::conv2d  (Conv2dOp)
Error:       DRAM Auto slice could not find valid slice configuration.
             Tried up to 2 slices for width-slicing on output dimension 44.
             Available L1: 1329888 bytes. Operation requires more memory
             than available even with maximum slicing.

Root cause layer: DepthNet ASPP, dilated conv with dilation=18.
  DepthNet (in_channels=512, mid_channels=512) receives the SECONDFPN output
  (6, 512, 16, 44) — all four FPN branches resampled to spatial 16x44 and
  concatenated. Inside DepthNet, the ASPP module runs four parallel dilated
  convolutions; the dilation=18 branch uses:

      nn.Conv2d(512, 512, kernel_size=3, padding=18, dilation=18, bias=False)
      input:  (6, 512, 16, 44)
      output: (6, 512, 16, 44)   <- output dimension 44 triggers the failure

  6 = number of BEVDepth camera views.
  16x44 = 256x704 input downsampled 16x through ResNet50 backbone
          (stem stride-2 + maxpool + layer2 stride-2 + layer3 stride-2).
"""

import os

import pytest
import torch
from infra import Framework
from infra.testers.single_chip.graph.graph_tester import run_graph_test

INPUTS_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bevdepth_depthnet_input.pt"
)


def _get_input() -> torch.Tensor:
    if os.path.exists(INPUTS_SAVE_PATH):
        return torch.load(INPUTS_SAVE_PATH, weights_only=True)
    x = torch.rand(6, 512, 16, 44, dtype=torch.float32)
    torch.save(x, INPUTS_SAVE_PATH)
    return x


@pytest.mark.single_device
def test_bevdepth_depthnet_aspp_dilation18(request):
    model = torch.nn.Conv2d(
        in_channels=512,
        out_channels=512,
        kernel_size=3,
        padding=18,
        dilation=18,
        bias=False,
    )
    run_graph_test(model, [_get_input()], framework=Framework.TORCH, request=request)
