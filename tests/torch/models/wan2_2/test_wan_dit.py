# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from .shared import RESOLUTIONS, load_dit, shard_dit_specs, wan22_mesh


class WanDiTWrapper(torch.nn.Module):
    """Return the velocity tensor from the diffusers output tuple."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


def test_wan_dit_480p():
    _run(resolution="480p", sharded=False)


def test_wan_dit_720p():
    _run(resolution="720p", sharded=False)


def test_wan_dit_480p_sharded():
    _run(resolution="480p", sharded=True)


def test_wan_dit_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    torch_xla.set_custom_compile_options(
        {"experimental-enable-dram-space-saving-optimization": True}
    )
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]
    t, h, w = shapes["latent_frames"], shapes["latent_h"], shapes["latent_w"]

    wrapper = WanDiTWrapper(load_dit()).eval().bfloat16()

    hidden_states = torch.randn(1, 48, t, h, w, dtype=torch.bfloat16)
    num_patches = t * (h // 2) * (w // 2)  # patchify stride (1, 2, 2)
    timestep = torch.full((1, num_patches), 500.0, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    mesh = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_dit_specs(m.dit)) if sharded else None

    run_graph_test(
        wrapper,
        [hidden_states, timestep, encoder_hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
