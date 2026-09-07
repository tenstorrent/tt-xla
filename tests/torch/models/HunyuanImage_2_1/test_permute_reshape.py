# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from loguru import logger

B, C, H, W = 1, 512, 1024, 1024  # conv output inside up_blocks[19]

class PermuteReshape(torch.nn.Module):
    """Permute + the hanging reshape; input is the 6D tensor after the first reshape."""

    def forward(self, h):
        h = h.permute(0, 3, 4, 1, 5, 2)
        return h.reshape(B, C // 4, H * 2, W * 2)


def test_permute_reshape():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    model = PermuteReshape().eval()
    # 6D tensor after the first reshape in HunyuanImageUpsample.forward
    # (decoder.up_blocks[19]): h.reshape(B, 2, 2, C // 4, H, W) with the conv
    # output h of shape 1x512x1024x1024 -> 1x2x2x128x1024x1024, bf16.
    inputs = [torch.randn(B, 2, 2, C // 4, H, W, dtype=torch.bfloat16)]
    logger.info("model={} input shape={} dtype={}", model, inputs[0].shape, inputs[0].dtype)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
