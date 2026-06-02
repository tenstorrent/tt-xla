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

from typing import Optional

import pytest
import torch
from infra import Framework, run_graph_test
from infra.utilities import Mesh

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import (
    RESOLUTIONS,
    VAEEncoderWrapper,
    load_vae,
    shard_vae_encoder_specs,
    wan22_mesh,
)

_COMPILER_CONFIG = CompilerConfig(
    optimization_level=1,
    enable_trace=True,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
def test_vae_encoder_720p_sharded():
    _run("720p", sharded=True)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
def test_vae_encoder_480p_sharded():
    _run("480p", sharded=True)


def _run(resolution: str, sharded: bool) -> None:
    shapes = RESOLUTIONS[resolution]
    torch.manual_seed(42)
    x = torch.randn(
        1,
        3,
        1,
        shapes["video_h"],
        shapes["video_w"],
        dtype=torch.bfloat16,
    )

    model = VAEEncoderWrapper(load_vae()).eval().bfloat16()

    mesh: Optional[Mesh] = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_vae_encoder_specs(m.vae)) if sharded else None

    run_graph_test(
        graph=model,
        inputs=[x],
        framework=Framework.TORCH,
        compiler_config=_COMPILER_CONFIG,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
