# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Krea Realtime — CausalWanModel (14B DiT) component test.

IN:  x        (1, 16, 3, 60, 104)  bfloat16   noisy latents
     t        (1, 3)               float32    timestep per latent frame
     context  (1, 512, 4096)       bfloat16   text embeddings
OUT: noise_pred (1, 16, 3, 60, 104) bfloat16

KV / cross-attention caches are built fresh inside the wrapper each forward,
so callers only need to supply tensor positionals (x, t, context).
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.torch.models.krea_realtime.shared import (
    DTYPE,
    KV_CACHE_SIZE,
    LATENT_H,
    LATENT_W,
    MAX_SEQ_LEN,
    NUM_CHANNELS_LATENTS,
    NUM_FRAMES_PER_BLOCK,
    NUM_LATENT_FRAMES,
    SEQ_LENGTH,
    TEXT_EMBED_DIM,
    krea_mesh,
    load_transformer,
    shard_transformer_specs,
)


def fixed_sinusoidal_embedding_1d(dim, position):
    """Device-agnostic replacement for Krea's CUDA-hardcoded version.

    Upstream: https://huggingface.co/krea/krea-realtime-video/blob/main/transformer/model.py#L33
    The original uses `device=torch.cuda.current_device()`, which crashes on
    CPU/TT. We replace it with `device=position.device`.
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, device=position.device, dtype=torch.float64).div(half),
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


class CausalWanWrapper(torch.nn.Module):
    """Wrap CausalWanModel so the forward signature is just (x, t, context).

    KV / cross-attention caches are rebuilt fresh each forward (zero-init,
    on the same device as the inputs). This keeps the test stateless and
    side-effect-free.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

        # Patch out the CUDA-hardcoded time embedding via the model's module namespace.
        transformer.forward.__globals__["sinusoidal_embedding_1d"] = (
            fixed_sinusoidal_embedding_1d
        )

        # Set the per-block self-attention attributes that WanRTSetupKVCache sets.
        for blk in self.transformer.blocks:
            blk.self_attn.local_attn_size = -1
            blk.self_attn.num_frame_per_block = NUM_FRAMES_PER_BLOCK

        # Cache shapes (computed once; tensors are built per-forward)
        self._num_blocks = len(self.transformer.blocks)
        self._num_heads = self.transformer.config.num_heads
        self._head_dim = self.transformer.config.dim // self._num_heads

    def _make_caches(self, device, dtype):
        kv_shape = [1, KV_CACHE_SIZE, self._num_heads, self._head_dim]
        ca_shape = [1, MAX_SEQ_LEN, self._num_heads, self._head_dim]
        kv_cache = [
            {
                "k": torch.zeros(kv_shape, dtype=dtype, device=device).contiguous(),
                "v": torch.zeros(kv_shape, dtype=dtype, device=device).contiguous(),
                "global_end_index": 0,
                "local_end_index": 0,
            }
            for _ in range(self._num_blocks)
        ]
        crossattn_cache = [
            {
                "k": torch.zeros(ca_shape, dtype=dtype, device=device).contiguous(),
                "v": torch.zeros(ca_shape, dtype=dtype, device=device).contiguous(),
                "is_init": False,
            }
            for _ in range(self._num_blocks)
        ]
        return kv_cache, crossattn_cache

    def forward(self, x, t, context):
        kv_cache, crossattn_cache = self._make_caches(x.device, x.dtype)
        return self.transformer(
            x=x,
            t=t,
            context=context,
            kv_cache=kv_cache,
            seq_len=SEQ_LENGTH,
            crossattn_cache=crossattn_cache,
            current_start=0,
            cache_start=None,
        )


@pytest.mark.skip(
    reason="OOM on single device — CausalWanModel exceeds single-chip memory; sharded variant runs"
)
def test_transformer():
    _run(sharded=False)


@pytest.mark.xfail(
    reason="Complex dtype (RoPE freqs from torch.polar) unsupported on TT — https://github.com/tenstorrent/tt-xla/issues/4464"
)
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    wrapper = CausalWanWrapper(load_transformer()).eval()

    x = torch.randn(
        1,
        NUM_CHANNELS_LATENTS,
        NUM_LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        dtype=DTYPE,
    )
    t = torch.full((1, NUM_FRAMES_PER_BLOCK), 1000.0, dtype=torch.float32)
    context = torch.randn(1, MAX_SEQ_LEN, TEXT_EMBED_DIM, dtype=DTYPE)

    mesh = krea_mesh() if sharded else None
    shard_spec_fn = (
        (lambda m: shard_transformer_specs(m.transformer)) if sharded else None
    )

    run_graph_test(
        wrapper,
        [x, t, context],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
