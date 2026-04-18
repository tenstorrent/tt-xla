# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — WanDiT (5B Transformer) component test.

The main compute bottleneck. Runs one DiT forward pass and compares
CPU vs TT output. Model config: 30 layers, 24 heads x 128 dim = 3072,
ffn_dim 14336, in/out 48 channels.

IN:  hidden_states (1, 48, latent_frames, latent_h, latent_w)
     timestep (1, num_patches)
     encoder_hidden_states (1, 512, 4096)
OUT: velocity (1, 48, latent_frames, latent_h, latent_w)
"""

import pytest
import torch

from .shared import RESOLUTIONS, compare_cpu_tt, load_dit, shard_dit_weights


class WanDiTWrapper(torch.nn.Module):
    """Return the velocity tensor from the diffusers output tuple."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


@pytest.mark.nightly
@pytest.mark.single_device
def test_wan_dit_480p():
    _run(resolution="480p", sharded=False)


@pytest.mark.nightly
@pytest.mark.single_device
def test_wan_dit_720p():
    _run(resolution="720p", sharded=False)


@pytest.mark.nightly
@pytest.mark.single_device
def test_wan_dit_480p_sharded():
    _run(resolution="480p", sharded=True)


@pytest.mark.nightly
@pytest.mark.single_device
def test_wan_dit_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]
    t, h, w = shapes["latent_frames"], shapes["latent_h"], shapes["latent_w"]

    wrapper = WanDiTWrapper(load_dit())

    hidden_states = torch.randn(1, 48, t, h, w)
    num_patches = t * (h // 2) * (w // 2)  # patchify stride (1, 2, 2)
    timestep = torch.full((1, num_patches), 500.0)
    encoder_hidden_states = torch.randn(1, 512, 4096)

    shard_fn = None
    if sharded:

        def shard_fn(model, mesh):
            shard_dit_weights(model.dit, mesh)

    compare_cpu_tt(
        wrapper,
        [hidden_states, timestep, encoder_hidden_states],
        shard_fn=shard_fn,
    )
