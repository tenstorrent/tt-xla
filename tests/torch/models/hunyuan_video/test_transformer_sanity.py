# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.single_chip.model.torch_model_tester import _mask_jax_accelerator


class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        # hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


@pytest.mark.nightly
@pytest.mark.model_test
def test_rope_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    model = (
        HunyuanVideoPatchEmbed(
            patch_size=(1, 2, 2),
            in_chans=16,
            embed_dim=3072,
        )
        .to(torch.bfloat16)
        .eval()
    )
    inputs = [
        torch.load(
            "/proj_sw/user_dev/mramanathan/whlb49_jun8_xla/tt-xla/hidden_states_hunyuan.pt"
        ),
    ]

    with _mask_jax_accelerator():
        run_graph_test(model, inputs, framework=Framework.TORCH)
