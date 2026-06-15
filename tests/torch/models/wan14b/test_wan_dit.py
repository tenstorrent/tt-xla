# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 A14B (I2V) — WanDiT (Transformer) component test.

Loads the I2V high-noise expert (``transformer/`` of the I2V-A14B repo) and
runs one forward pass, comparing CPU vs TT output. Model config: 40 layers,
40 heads x 128 dim = 5120, ffn_dim 13824, in_channels=36, out_channels=16.

A14B-I2V conditions on the image purely through the 36 concatenated input
channels (16 latent + 4 mask + 16 image-condition); its config has
``image_dim=None``/``added_kv_proj_dim=None`` (no CLIP image cross-attention),
and the i2v pipeline rejects ``image_embeds`` for boundary-ratio models. So the
forward signature, scalar timestep, text ``encoder_hidden_states`` and sharding
are all standard — only ``hidden_states`` carries the 36 conditioning channels.

A14B uses ``expand_timesteps=False`` — timestep is a scalar per-batch tensor,
not per-token like the 5B TI2V model.

IN:  hidden_states (1, 36, latent_frames, latent_h, latent_w)
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
    _patch_wan_time_embedder_dtype_probe,
    torch_function_override_disabled,
)
from .shared import (
    MODEL_ID_I2V,
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


@pytest.mark.xfail(
    reason="PCC comparison fails on the full DiT (DiT sharding limitation)"
)
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
@pytest.mark.skip(reason="Skipping due to the long running time: > 40 minutes")
def test_wan_dit_720p_sharded():
    _run("720p", sharded=True)


@pytest.mark.xfail(
    reason="PCC comparison fails on the full DiT (DiT sharding limitation)"
)
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

    shapes = RESOLUTIONS[resolution]
    t, h, w = shapes["latent_frames"], shapes["latent_h"], shapes["latent_w"]

    model = (
        WanDiTWrapper(load_dit(max_blocks=MAX_BLOCKS, model_id=MODEL_ID_I2V))
        .eval()
        .bfloat16()
    )
    # I2V-A14B DiT takes 36 input channels (16 latent + 4 mask + 16 image-cond)
    in_channels = model.dit.config.in_channels

    torch.manual_seed(42)
    hidden_states = torch.randn(1, in_channels, t, h, w, dtype=torch.bfloat16)
    timestep = torch.full((1,), 500.0, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    mesh: Optional[Mesh] = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_dit_specs(m.dit)) if sharded else None

    # SP activation sharding is the companion to shard_dit_specs (weights): its
    # forward hooks constrain how Shardy propagates shards through the RoPE /
    # block-entry reshapes. Registered here on the model so the hooks survive
    # the runner's model.to(device) move.
    if sharded:
        apply_dit_sp_activation_sharding(model.dit, mesh)

    # Pop the tt_torch matmul/linear override for the scope so the lazy trace
    # inside run_graph_test sees the RoPE unflatten without it; restored on exit
    # (unlike the bare pop, this doesn't leak into other tests in the session).
    with torch_function_override_disabled():
        run_graph_test(
            graph=model,
            inputs=[hidden_states, timestep, encoder_hidden_states],
            framework=Framework.TORCH,
            compiler_config=_COMPILER_CONFIG,
            mesh=mesh,
            shard_spec_fn=shard_spec_fn,
        )
