# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
krea-realtime-video — CausalWanModel (14.3B real-time DiT) component test.

The main compute + memory bottleneck. ~14.3B params (28.6 GB bf16) plus a
per-block KV cache, so it is weight-bound on every single chip — the unsharded
nodes OOM by design; the sharded nodes are the tensor-parallel bring-up target.

This is a CausVid-style causal model: it denoises `num_frames_per_block` (3)
latent frames per realtime step using mutable per-block KV / cross-attention
caches. KreaCausalDiTWrapper builds those caches and adapts the model's
list-of-tensors I/O to a tensors-only forward (one realtime step, frame 0).

Config: dim 5120, 40 layers, 40 heads x 128, ffn_dim 13824, in/out 16 channels.

IN:  latents (1, 16, 3, latent_h, latent_w)
     timestep (1, 3)
     context  (1, 512, 4096)        # UMT5-XXL text embeddings
OUT: noise_pred (1, 16, 3, latent_h, latent_w)
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import (
    RESOLUTIONS,
    TRANSFORMER_CFG,
    KreaCausalDiTWrapper,
    krea_mesh,
    load_transformer,
    shard_causal_dit_specs,
)


def test_causal_wan_dit_480p():  # weight-bound (OOM) on a single device
    _run(resolution="480p", sharded=False)


def test_causal_wan_dit_480p_sharded():
    _run(resolution="480p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    compiler_config = CompilerConfig(optimization_level=1)
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]
    cfg = TRANSFORMER_CFG
    f = cfg["num_frames_per_block"]

    wrapper = KreaCausalDiTWrapper(load_transformer()).eval().bfloat16()

    latents = torch.randn(
        1,
        cfg["in_dim"],
        f,
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.bfloat16,
    )
    timestep = torch.full((1, f), 500.0, dtype=torch.bfloat16)
    context = torch.randn(1, cfg["text_len"], cfg["text_dim"], dtype=torch.bfloat16)

    mesh = krea_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_causal_dit_specs(m.dit)) if sharded else None

    run_graph_test(
        wrapper,
        [latents, timestep, context],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
