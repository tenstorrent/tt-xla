# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category, failed_runtime


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


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    "in_channels,out_channels",
    [
        pytest.param(
            768,
            768,
            id="in_and_out_div32",
            marks=pytest.mark.xfail(
                reason=failed_runtime(
                    "L1 OOM: https://github.com/tenstorrent/tt-xla/issues/3108"
                )
            ),
        ),
        pytest.param(12, 512, id="in_not_div32"),
        pytest.param(
            512,
            12,
            id="out_not_div32",
            marks=pytest.mark.xfail(
                reason=failed_runtime(
                    "L1 OOM: https://github.com/tenstorrent/tt-xla/issues/3108"
                )
            ),
        ),
        pytest.param(12, 34, id="neither_div32"),
    ],
)
def test_mochi_causal_conv3d(in_channels, out_channels):
    """
    Test CogVideoXCausalConv3d with channels divisible by 32 or not.
    """
    model = MochiCausalConv3dWrapper(
        in_channels=in_channels, out_channels=out_channels, kernel_size=3
    )

    run_graph_test_with_random_inputs(
        model,
        [(1, in_channels, 4, 60, 106)],
        framework=Framework.TORCH,
    )
