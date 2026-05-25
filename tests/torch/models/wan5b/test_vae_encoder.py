# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — 3D Causal VAE Encoder component test.

Encodes a single-frame image into the 48-channel latent used for I2V
conditioning.

IN:  (1, 3, 1, video_h, video_w) float
OUT: latent_dist.mean (1, 48, 1, latent_h, latent_w)
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import RESOLUTIONS, load_vae, shard_vae_encoder_specs, wan22_mesh


class VAEEncoderWrapper(torch.nn.Module):
    """Run encoder and return the deterministic mean latent."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(x).latent_dist.mean


def test_vae_encoder_480p():
    _run(resolution="480p", sharded=False)


def test_vae_encoder_720p():
    _run(resolution="720p", sharded=False)


def test_vae_encoder_480p_sharded():
    _run(resolution="480p", sharded=True)


def test_vae_encoder_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    compiler_config = CompilerConfig(optimization_level=1)
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]

    wrapper = VAEEncoderWrapper(load_vae()).eval().bfloat16()

    # Single-frame image encoding
    x = torch.randn(1, 3, 1, shapes["video_h"], shapes["video_w"], dtype=torch.bfloat16)

    mesh = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_vae_encoder_specs(m.vae)) if sharded else None

    run_graph_test(
        wrapper,
        [x],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
