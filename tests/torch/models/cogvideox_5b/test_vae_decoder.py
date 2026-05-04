# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CogVideoX-5b — AutoencoderKLCogVideoX (3D causal VAE) decoder component test.

IN:  z       (1, 16, 3, 60, 90)   bfloat16   permuted latents
                                            (batch, channels, num_latent_frames, h, w)
OUT: sample  (1, 3, 9, 480, 720)  bfloat16   reconstructed video

Shapes match the modified inference at tests/torch/models/test_cog5x_num1.py
(num_frames=9, height=480, width=720). The pipeline permutes latents from
(batch, num_latent_frames, channels, h, w) to (batch, channels,
num_latent_frames, h, w) before decoding.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.torch.models.cogvideox_5b.shared import (
    DTYPE,
    LATENT_H,
    LATENT_W,
    NUM_CHANNELS_LATENTS,
    NUM_LATENT_FRAMES,
    cogvideox_mesh,
    load_vae,
    shard_vae_decoder_specs,
)


class VAEDecoderWrapper(torch.nn.Module):
    """Wrap AutoencoderKLCogVideoX so the forward signature is just (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


@pytest.mark.nightly
@pytest.mark.model_test
def test_vae_decoder():
    _run(sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
def test_vae_decoder_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    wrapper = VAEDecoderWrapper(load_vae()).eval()

    z = torch.randn(
        1,
        NUM_CHANNELS_LATENTS,
        NUM_LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        dtype=DTYPE,
    )

    mesh = cogvideox_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_vae_decoder_specs(m.vae)) if sharded else None

    run_graph_test(
        wrapper,
        [z],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
