# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 A14B — 3D Causal VAE Decoder component test (Wan 2.1 VAE).

Decodes a denoised latent back to pixel space.

IN:  z (1, 16, latent_frames, latent_h, latent_w)
OUT: sample (1, 3, num_frames, video_h, video_w)
"""

from typing import Optional

import pytest
import torch
from infra import ComparisonConfig, Framework, run_graph_test
from infra.evaluators import PccConfig
from infra.utilities import Mesh

from tests.infra.testers.compiler_config import CompilerConfig

from .monkey_patch import _patch_wan_resample_rep_sentinel, safe_xla_slicing
from .shared import (
    LATENT_CHANNELS,
    RESOLUTIONS,
    VAEDecoderWrapper,
    load_vae,
    shard_vae_decoder_specs,
    wan22_mesh,
)

_COMPILER_CONFIG = CompilerConfig(
    optimization_level=1,
    experimental_enable_dram_space_saving_optimization=True,
    export_path="model",
    export_model_name="vae_decoder",
    enable_trace=True,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
@pytest.mark.skip(
    reason="currently slow so skipping for now: we need to set proper config for conv3d in tt-mlir"
)
def test_vae_decoder_720p():
    _run("720p", sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
@pytest.mark.skip(
    reason="currently slow so skipping for now: we need to set proper config for conv3d in tt-mlir"
)
def test_vae_decoder_480p():
    _run("480p", sharded=False)


def _run(resolution: str, sharded: bool) -> None:
    # Apply monkey patches here (not at module top) so they don't leak into
    # other tests collected in the same pytest session.
    _patch_wan_resample_rep_sentinel()

    shapes = RESOLUTIONS[resolution]
    torch.manual_seed(42)
    z = torch.randn(
        1,
        LATENT_CHANNELS,
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.bfloat16,
    )

    model = VAEDecoderWrapper(load_vae()).eval().bfloat16()

    mesh: Optional[Mesh] = wan22_mesh() if sharded else None
    shard_spec_fn = (
        (lambda m: shard_vae_decoder_specs(m.vae, mesh)) if sharded else None
    )

    # safe_xla_slicing wraps the entire run: its TorchFunctionMode stays on
    # the stack across CPU golden, dynamo trace + compile, and TT execution.
    # Required because AutoencoderKLWan relies on CPU's silent slice-clamping
    # (e.g. x[:, :, -2:, :, :] on a size-1 temporal dim), which torch-xla
    # rejects unless we normalize the indices first.
    with safe_xla_slicing():
        run_graph_test(
            graph=model,
            inputs=[z],
            framework=Framework.TORCH,
            compiler_config=_COMPILER_CONFIG,
            mesh=mesh,
            shard_spec_fn=shard_spec_fn,
            comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        )
