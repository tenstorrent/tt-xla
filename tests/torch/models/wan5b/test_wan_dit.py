# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — WanDiT (5B Transformer) component test.

The main compute bottleneck. Runs one DiT forward pass and compares
CPU vs TT output. Model config: 30 layers, 24 heads x 128 dim = 3072,
ffn_dim 14336, in/out 48 channels.

IN:  hidden_states (1, 48, latent_frames, latent_h, latent_w)
     timestep (1, num_patches)
     encoder_hidden_states (1, 512, 4096)
OUT: velocity (1, 48, latent_frames, latent_h, latent_w)
"""

from typing import Optional

import pytest
import torch
from infra import Framework, run_graph_test
from infra.utilities import Mesh

from tests.infra.testers.compiler_config import CompilerConfig

from .monkey_patch import _disable_tt_torch_function_override, _patch_apply_lora_scale
from .shared import RESOLUTIONS, WanDiTWrapper, load_dit, shard_dit_specs, wan22_mesh

# Module-load patches — must run before model load and torch.compile.
_patch_apply_lora_scale()
_disable_tt_torch_function_override()


# Set to 0 to run the full model, otherwise set to the number of blocks to run.
MAX_BLOCKS = 0


_COMPILER_CONFIG = CompilerConfig(
    optimization_level=1,
    experimental_enable_dram_space_saving_optimization=True,
    enable_trace=True,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
def test_wan_dit_480p_sharded():
    _run("480p", sharded=True)


def _run(resolution: str, sharded: bool) -> None:
    shapes = RESOLUTIONS[resolution]
    t, h, w = shapes["latent_frames"], shapes["latent_h"], shapes["latent_w"]

    torch.manual_seed(42)
    hidden_states = torch.randn(1, 48, t, h, w, dtype=torch.bfloat16)
    num_patches = t * (h // 2) * (w // 2)  # patchify stride (1, 2, 2)
    timestep = torch.full((1, num_patches), 500.0, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    model = WanDiTWrapper(load_dit(max_blocks=MAX_BLOCKS)).eval().bfloat16()

    mesh: Optional[Mesh] = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_dit_specs(m.dit)) if sharded else None

    run_graph_test(
        graph=model,
        inputs=[hidden_states, timestep, encoder_hidden_states],
        framework=Framework.TORCH,
        compiler_config=_COMPILER_CONFIG,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
