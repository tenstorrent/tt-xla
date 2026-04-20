# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — 3D Causal VAE Decoder component test.

Decodes a denoised latent back to pixel space.

IN:  z (1, 48, latent_frames, latent_h, latent_w)
OUT: sample (1, 3, num_frames, video_h, video_w)
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from .shared import RESOLUTIONS, load_vae, shard_vae_decoder_specs, wan22_mesh


class VAEDecoderWrapper(torch.nn.Module):
    """Run decoder and return the reconstructed sample tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


def test_vae_decoder_480p():
    _run(resolution="480p", sharded=False)


def test_vae_decoder_720p():
    _run(resolution="720p", sharded=False)


def test_vae_decoder_480p_sharded():
    _run(resolution="480p", sharded=True)


def test_vae_decoder_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]

    wrapper = VAEDecoderWrapper(load_vae()).eval().bfloat16()

    z = torch.randn(
        1,
        48,
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.bfloat16,
    )

    mesh = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_vae_decoder_specs(m.vae)) if sharded else None

    run_graph_test(
        wrapper,
        [z],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
