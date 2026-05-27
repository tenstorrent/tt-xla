import torch

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo — AutoencoderKLHunyuanVideo decoder component test."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.single_chip.model.torch_model_tester import _mask_jax_accelerator


def get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray | int,
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (
            theta
            ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
        )
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = (
            freqs.cos()
            .repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
            .float()
        )  # [S, D]
        freqs_sin = (
            freqs.sin()
            .repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
            .float()
        )  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis


class Lumina2RotaryPosEmbed(nn.Module):
    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        axes_lens: list[int] = (300, 512, 512),
        patch_size: int = 2,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

        self.freqs_cis = self._precompute_freqs_cis(axes_dim, axes_lens, theta)

    def _precompute_freqs_cis(
        self, axes_dim: list[int], axes_lens: list[int], theta: int
    ) -> list[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )
        freqs_dtype = torch.float32
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(
                d, e, theta=self.theta, freqs_dtype=freqs_dtype
            )
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = self.freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(
                torch.gather(
                    freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index
                )
            )
        return torch.cat(result, dim=-1).to(device)

    def forward(self, position_ids: torch.Tensor):
        freqs_cis = self._get_freqs_cis(position_ids)
        return freqs_cis


@pytest.mark.nightly
@pytest.mark.model_test
def test_vae_decoder_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    model = (
        Lumina2RotaryPosEmbed(
            theta=10000, axes_dim=[32, 32, 32], axes_lens=[300, 512, 512], patch_size=2
        )
        .to(torch.bfloat16)
        .eval()
    )
    inputs = [torch.load("./position_ids_lumina.pt")]

    with _mask_jax_accelerator():
        run_graph_test(model, inputs, framework=Framework.TORCH)
