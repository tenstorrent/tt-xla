# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo — AutoencoderKLHunyuanVideo decoder component test."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.single_chip.model.torch_model_tester import _mask_jax_accelerator


class HunyuanVideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )

        self.pad_mode = pad_mode
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(
            hidden_states, self.time_causal_padding, mode=self.pad_mode
        )
        return hidden_states


@pytest.mark.nightly
@pytest.mark.model_test
def test_vae_decoder_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    model = (
        HunyuanVideoCausalConv3d(
            in_channels=16, out_channels=512, kernel_size=3, stride=1, padding=0
        )
        .to(torch.bfloat16)
        .eval()
    )
    inputs = [
        torch.load("tests/torch/models/hunyuan_videohidden_states_vae.pt").to(
            torch.bfloat16
        )
    ]

    with _mask_jax_accelerator():
        run_graph_test(model, inputs, framework=Framework.TORCH)
