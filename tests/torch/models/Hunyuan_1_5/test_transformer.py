# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HunyuanVideo 1.5 — HunyuanVideo15Transformer3DModel (DiT) component test.

Size: 8.33B params

IN:  hidden_states           (1, 65, 5, 30, 53)   bfloat16  (32 latent + 32 cond + 1 mask)
     timestep                (1,)                 bfloat16
     encoder_hidden_states   (1, 1000, 3584)      bfloat16  (Qwen embeds)
     encoder_attention_mask  (1, 1000)            bfloat16
     encoder_hidden_states_2 (1, 256, 1472)       bfloat16  (ByT5 embeds)
     encoder_attention_mask_2(1, 256)             bfloat16
     image_embeds            (1, N, 1152)         bfloat16  (zeros for t2v)
OUT: noise_pred              (1, 32, 5, 30, 53)   bfloat16

In the pipeline only the transformer's hidden_states output is used; the
wrapper extracts that tensor (return_dict=False, then [0]) so it returns
a plain tensor instead of a model output object.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.torch.models.Hunyuan_1_5.shared import (
    DTYPE,
    IMAGE_EMBED_DIM,
    IMAGE_EMBED_SEQ,
    LATENT_H,
    LATENT_W,
    NUM_LATENT_FRAMES,
    TEXT_EMBED_2_DIM,
    TEXT_EMBED_DIM,
    TRANSFORMER_IN_CHANNELS,
    TRANSFORMER_TEXT_2_SEQ,
    TRANSFORMER_TEXT_SEQ,
    hunyuan_mesh,
    load_transformer,
    shard_transformer_specs,
)


class HunyuanVideo15TransformerWrapper(torch.nn.Module):
    """Wrap HunyuanVideo15Transformer3DModel with a tensor-only forward."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        image_embeds,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            image_embeds=image_embeds,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states_2=encoder_hidden_states_2,
            encoder_attention_mask_2=encoder_attention_mask_2,
            return_dict=False,
        )[0]


@pytest.mark.skip(
    reason="model size > 8B — won't fit on a single chip; sharded variant runs"
)
def test_transformer():
    _run(sharded=False)


@pytest.mark.xfail(
    reason="SPMD compilation gives trivial mesh size [1,1] -> 'Device count mismatch: 2 vs 1' at execute — https://github.com/tenstorrent/tt-xla/issues/4486"
)
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    wrapper = HunyuanVideo15TransformerWrapper(load_transformer()).eval()

    hidden_states = torch.randn(
        1,
        TRANSFORMER_IN_CHANNELS,
        NUM_LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        dtype=DTYPE,
    )
    timestep = torch.tensor([1000.0], dtype=DTYPE)
    encoder_hidden_states = torch.randn(
        1, TRANSFORMER_TEXT_SEQ, TEXT_EMBED_DIM, dtype=DTYPE
    )
    encoder_attention_mask = torch.ones(1, TRANSFORMER_TEXT_SEQ, dtype=DTYPE)
    encoder_hidden_states_2 = torch.randn(
        1, TRANSFORMER_TEXT_2_SEQ, TEXT_EMBED_2_DIM, dtype=DTYPE
    )
    encoder_attention_mask_2 = torch.ones(1, TRANSFORMER_TEXT_2_SEQ, dtype=DTYPE)
    image_embeds = torch.zeros(1, IMAGE_EMBED_SEQ, IMAGE_EMBED_DIM, dtype=DTYPE)

    mesh = hunyuan_mesh() if sharded else None
    shard_spec_fn = (
        (lambda m: shard_transformer_specs(m.transformer)) if sharded else None
    )

    run_graph_test(
        wrapper,
        [
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            image_embeds,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
