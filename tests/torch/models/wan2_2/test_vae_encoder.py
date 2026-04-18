# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — 3D Causal VAE Encoder component test.

Encodes a single-frame image into the 48-channel latent used for I2V
conditioning.

IN:  (1, 3, 1, video_h, video_w) float
OUT: latent_dist.mean (1, 48, 1, latent_h, latent_w)
"""

import pytest
import torch

from .shared import RESOLUTIONS, compare_cpu_tt, load_vae, shard_vae_encoder_weights


class VAEEncoderWrapper(torch.nn.Module):
    """Run encoder and return the deterministic mean latent."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(x).latent_dist.mean


@pytest.mark.nightly
@pytest.mark.single_device
def test_vae_encoder_480p():
    _run(resolution="480p", sharded=False)


@pytest.mark.nightly
@pytest.mark.single_device
def test_vae_encoder_720p():
    _run(resolution="720p", sharded=False)


@pytest.mark.nightly
@pytest.mark.single_device
def test_vae_encoder_480p_sharded():
    _run(resolution="480p", sharded=True)


@pytest.mark.nightly
@pytest.mark.single_device
def test_vae_encoder_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]
    wrapper = VAEEncoderWrapper(load_vae())

    # Single-frame image encoding (the I2V use case)
    x = torch.randn(1, 3, 1, shapes["video_h"], shapes["video_w"], dtype=torch.bfloat16)

    shard_fn = None
    if sharded:

        def shard_fn(model, mesh):
            shard_vae_encoder_weights(model.vae, mesh)

    compare_cpu_tt(wrapper, [x], shard_fn=shard_fn)
