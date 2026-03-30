# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test reproducing the Conv3d OOM from qwen_2_5_vl/pytorch-3B_Instruct.

Failing op:  ttnn::experimental::conv3d  (Conv3dDeviceOperation)
Error:       Statically allocated circular buffers grow to 1738528 B
             which is beyond max L1 size of 1499136 B

Qwen2_5_VisionPatchEmbed uses:
    nn.Conv3d(in_channels=3, out_channels=1280,
              kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
Input shape to Conv3d: (N, 3, 2, 14, 14)
N=1024 corresponds to a 448x448 image (32x32 spatial patches, 1 temporal group).
"""

import os

import pytest
import torch
from infra import Framework
from infra.testers.single_chip.graph.graph_tester import run_graph_test

INPUTS_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qwen_patch_embed_pixel_values.pt"
)


def _get_input() -> torch.Tensor:
    if os.path.exists(INPUTS_SAVE_PATH):
        return torch.load(INPUTS_SAVE_PATH, weights_only=True)
    x = torch.rand(1024, 3, 2, 14, 14, dtype=torch.float32)
    torch.save(x, INPUTS_SAVE_PATH)
    return x


@pytest.mark.single_device
def test_qwen_2_5_vl_patch_embed_conv3d(request):
    model = torch.nn.Conv3d(
        in_channels=3,
        out_channels=1280,
        kernel_size=(2, 14, 14),
        stride=(2, 14, 14),
        bias=False,
    )
    run_graph_test(model, [_get_input()], framework=Framework.TORCH, request=request)
