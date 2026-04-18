# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — 3D Causal VAE Decoder component test.

Decodes a denoised latent back to pixel space.

IN:  z (1, 48, latent_frames, latent_h, latent_w)
OUT: sample (1, 3, num_frames, video_h, video_w)
"""

import pytest
import torch

from .shared import RESOLUTIONS, compare_cpu_tt, load_vae, shard_vae_decoder_weights


class VAEDecoderWrapper(torch.nn.Module):
    """Run decoder and return the reconstructed sample tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


@pytest.mark.nightly
@pytest.mark.single_device
def test_vae_decoder_480p():
    _run(resolution="480p", sharded=False)


@pytest.mark.nightly
@pytest.mark.single_device
def test_vae_decoder_720p():
    _run(resolution="720p", sharded=False)


@pytest.mark.nightly
@pytest.mark.single_device
def test_vae_decoder_480p_sharded():
    _run(resolution="480p", sharded=True)


@pytest.mark.nightly
@pytest.mark.single_device
def test_vae_decoder_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]
    wrapper = VAEDecoderWrapper(load_vae())

    z = torch.randn(
        1,
        48,
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.bfloat16,
    )

    shard_fn = None
    if sharded:

        def shard_fn(model, mesh):
            shard_vae_decoder_weights(model.vae, mesh)

    compare_cpu_tt(wrapper, [z], shard_fn=shard_fn)
