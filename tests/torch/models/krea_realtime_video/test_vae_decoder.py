# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
krea-realtime-video — AutoencoderKLWan (3D Causal VAE) Decoder component test.

Same VAE class as Wan 2.1/2.2 (z_dim 16, 8x spatial / 4x temporal). Decodes a
denoised latent back to pixel space. This is the small component (~0.1-0.3B) and
is expected to fit a single device — the unsharded nodes are the mergeable pass.

IN:  z (1, 16, latent_frames, latent_h, latent_w)
OUT: sample (1, 3, num_frames, video_h, video_w)
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import RESOLUTIONS, krea_mesh, load_vae, shard_vae_decoder_specs


class VAEDecoderWrapper(torch.nn.Module):
    """Run decoder and return the reconstructed sample tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


def test_vae_decoder_480p():
    _run(resolution="480p", sharded=False)


def test_vae_decoder_480p_sharded():
    _run(resolution="480p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    compiler_config = CompilerConfig(optimization_level=1)
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]

    wrapper = VAEDecoderWrapper(load_vae()).eval().bfloat16()

    z = torch.randn(
        1,
        shapes["latent_c"],
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.bfloat16,
    )

    mesh = krea_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_vae_decoder_specs(m.vae)) if sharded else None

    run_graph_test(
        wrapper,
        [z],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
