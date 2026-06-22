# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 A14B — WanDiT (Transformer) component test.

Loads the high-noise expert (``transformer/``) and runs one forward pass,
comparing CPU vs TT output. Model config: 40 layers, 40 heads x 128 dim =
5120, ffn_dim 13824, in/out 16 channels (T2V-A14B; I2V variant uses
in_channels=36).

A14B uses ``expand_timesteps=False`` — timestep is a scalar per-batch
tensor, not per-token like the 5B TI2V model.

IN:  hidden_states (1, 16, latent_frames, latent_h, latent_w)
     timestep (1,)
     encoder_hidden_states (1, 512, 4096)
OUT: velocity (1, 16, latent_frames, latent_h, latent_w)
"""

from typing import Optional

import pytest
import torch
from infra import Framework, run_graph_test
from infra.utilities import Mesh

from tests.infra.testers.compiler_config import CompilerConfig

from .monkey_patch import (
    _disable_tt_torch_function_override,
    _patch_wan_time_embedder_dtype_probe,
)
from .shared import (
    LATENT_CHANNELS,
    RESOLUTIONS,
    WanDiTWrapper,
    apply_dit_sp_activation_sharding,
    load_dit,
    shard_dit_specs,
    wan22_mesh,
)

# Set to 0 to run the full model, otherwise set to the number of blocks to run.
MAX_BLOCKS = 0

_COMPILER_CONFIG = CompilerConfig(
    optimization_level=1,
    experimental_enable_dram_space_saving_optimization=True,
    enable_trace=True,
)


@pytest.mark.xfail(reason="PCC comparison fails: captured 0.75 on single decoder block")
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
# @pytest.mark.skip(reason="Skipping due to the long running time: > 40 minutes")
def test_wan_dit_720p_sharded():
    _run("720p", sharded=True)


@pytest.mark.xfail(reason="PCC comparison fails: captured 0.71")
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
def test_wan_dit_480p_sharded():
    _run("480p", sharded=True)


def _run(resolution: str, sharded: bool) -> None:
    # Apply monkey patches here (not at module top) so they don't leak into
    # other tests collected in the same pytest session.
    # torch-2.10 dynamo workaround: rewrite the .parameters() dtype probe in
    # WanTimeTextImageEmbedding.forward (the accelerator-stub and XLA guard
    # shims it also relies on are applied globally by tt_torch's package
    # patches in tt_torch/__init__.py).
    _patch_wan_time_embedder_dtype_probe()
    # diffusers Wan-transformer rewrites (perf + tracing).
    _disable_tt_torch_function_override()

    shapes = RESOLUTIONS[resolution]
    t, h, w = shapes["latent_frames"], shapes["latent_h"], shapes["latent_w"]

    torch.manual_seed(42)
    hidden_states = torch.randn(1, LATENT_CHANNELS, t, h, w, dtype=torch.bfloat16)
    timestep = torch.full((1,), 500.0, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    model = WanDiTWrapper(load_dit(max_blocks=MAX_BLOCKS)).eval().bfloat16()

    mesh: Optional[Mesh] = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_dit_specs(m.dit)) if sharded else None

    # SP activation sharding is the companion to shard_dit_specs (weights): its
    # forward hooks constrain how Shardy propagates shards through the RoPE /
    # block-entry reshapes. Registered here on the model so the hooks survive
    # the runner's model.to(device) move. The tt_torch matmul/linear override is
    # already popped above via _disable_tt_torch_function_override(), so the lazy
    # trace inside run_graph_test sees the RoPE unflatten without it.
    if sharded:
        apply_dit_sp_activation_sharding(model.dit, mesh)

    run_graph_test(
        graph=model,
        inputs=[hidden_states, timestep, encoder_hidden_states],
        framework=Framework.TORCH,
        compiler_config=_COMPILER_CONFIG,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
