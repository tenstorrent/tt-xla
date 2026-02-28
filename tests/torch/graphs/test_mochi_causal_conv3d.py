# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d
from infra import Framework, run_graph_test_with_random_inputs
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
    """
    model = MochiCausalConv3dWrapper(in_channels=768, out_channels=768, kernel_size=3)

    run_graph_test_with_random_inputs(
        model,
        [(1, 768, 4, 60, 106)],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    "in_channels,out_channels",
    [
        pytest.param(12, 512, id="in_not_div32"),
        pytest.param(512, 12, id="out_not_div32"),
        pytest.param(12, 34, id="neither_div32"),
    ],
)
def test_mochi_causal_conv3d_non_aligned_channels(in_channels, out_channels):
    """
    Test CogVideoXCausalConv3d with channels not divisible by 32.

    TT-Metal conv3d requires channel alignment to 32. These cases test
    that padding/alignment is handled correctly when in_channels,
    out_channels, or both are not multiples of 32.
    """
    model = MochiCausalConv3dWrapper(
        in_channels=in_channels, out_channels=out_channels, kernel_size=3
    )

    run_graph_test_with_random_inputs(
        model,
        [(1, in_channels, 4, 60, 106)],
        framework=Framework.TORCH,
    )
