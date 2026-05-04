# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CogVideoX-5b — CogVideoXTransformer3DModel (5B DiT) component test.

IN:  hidden_states         (1, 3, 16, 60, 90)   bfloat16   noisy latents
                                                          (batch, num_latent_frames,
                                                           channels, h, w)
     timestep              (1,)                 int64      one denoising step
     encoder_hidden_states (1, 226, 4096)       bfloat16   T5 text embeddings
OUT: noise_pred            (1, 3, 16, 60, 90)   bfloat16

Shapes match the modified inference at tests/torch/models/test_cog5x_num1.py
(num_frames=9, height=480, width=720). CogVideoX-5b uses additive sin/cos
positional embeddings (use_rotary_positional_embeddings=False), so
image_rotary_emb is not provided.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.torch.models.cogvideox_5b.shared import (
    DTYPE,
    LATENT_H,
    LATENT_W,
    MAX_SEQ_LEN,
    NUM_CHANNELS_LATENTS,
    NUM_LATENT_FRAMES,
    TEXT_EMBED_DIM,
    cogvideox_mesh,
    load_transformer,
    shard_transformer_specs,
)


class CogVideoXTransformerWrapper(torch.nn.Module):
    """Wrap CogVideoXTransformer3DModel so the forward signature is
    (hidden_states, timestep, encoder_hidden_states) and the return is a
    plain tensor.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )[0]


def test_transformer():
    _run(sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
def test_transformer_sharded():
    _run(sharded=True)


@pytest.mark.push
def test_transformer_single_block_sharded():
    """Smoke test: truncate to a single transformer block for compile-time
    debugging (matches the spirit of test_cog5x_num1.py's reduced run)."""
    _run(sharded=True, max_blocks=1)


def _run(sharded: bool, max_blocks: int = 0):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    wrapper = CogVideoXTransformerWrapper(
        load_transformer(max_blocks=max_blocks)
    ).eval()

    hidden_states = torch.randn(
        1,
        NUM_LATENT_FRAMES,
        NUM_CHANNELS_LATENTS,
        LATENT_H,
        LATENT_W,
        dtype=DTYPE,
    )
    timestep = torch.tensor([1000], dtype=torch.int64)
    encoder_hidden_states = torch.randn(1, MAX_SEQ_LEN, TEXT_EMBED_DIM, dtype=DTYPE)

    mesh = cogvideox_mesh() if sharded else None
    shard_spec_fn = (
        (lambda m: shard_transformer_specs(m.transformer)) if sharded else None
    )

    run_graph_test(
        wrapper,
        [hidden_states, timestep, encoder_hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
