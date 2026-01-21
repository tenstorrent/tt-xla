# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d
from utils import Category


class MochiCausalConv3dWrapper(torch.nn.Module):
    """
    Wrapper for CogVideoXCausalConv3d that handles the tuple output.

    CogVideoXCausalConv3d returns (output, conv_cache) tuple, but for testing
    we only need the output tensor.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = CogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            pad_mode="replicate",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _conv_cache = self.conv(x)
        return output


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_mochi_causal_conv3d():
    """
    Test CogVideoXCausalConv3d from diffusers library with Mochi VAE settings.

    This tests the causal 3D convolution block used in Mochi video generation model.
    Settings:
    - in_channels=768, out_channels=768
    - kernel_size=3, stride=1
    - pad_mode="replicate" (causal padding for temporal dimension)

    Input shape matches Mochi decoder: [B, C, T, H, W] = [1, 768, 4, 60, 106]

    Note: We skip comparison with CPU since CPU doesn't support bfloat16 and
    we need bfloat16 for conv3d op on TT device.
    """

    xr.set_device_type("TT")
    device = torch_xla.device()

    model = MochiCausalConv3dWrapper(
        in_channels=768, out_channels=768, kernel_size=3
    ).to(device=device, dtype=torch.bfloat16)

    model = torch.compile(model, backend="tt")

    input_tensor = torch.randn(1, 768, 4, 60, 106, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        output = model(input_tensor)
        torch_xla.sync()
        assert output.shape == (
            1,
            768,
            4,
            60,
            106,
        ), f"Output shape mismatch, expected (1, 768, 4, 60, 106) but got {output.shape}"
