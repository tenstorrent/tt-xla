# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — UMT5-XXL Text Encoder component test.

IN:  input_ids (1, 512) int64, attention_mask (1, 512) int64
OUT: last_hidden_state (1, 512, 4096) float
"""

import pytest
import torch

from .shared import RESOLUTIONS, compare_cpu_tt, load_umt5, shard_umt5_weights


class UMT5Wrapper(torch.nn.Module):
    """Return last_hidden_state as a plain tensor (not a model output object)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


@pytest.mark.nightly
@pytest.mark.single_device
def test_umt5_480p():
    _run(resolution="480p", sharded=False)


@pytest.mark.nightly
@pytest.mark.single_device
def test_umt5_720p():
    _run(resolution="720p", sharded=False)


@pytest.mark.nightly
@pytest.mark.single_device
def test_umt5_480p_sharded():
    _run(resolution="480p", sharded=True)


@pytest.mark.nightly
@pytest.mark.single_device
def test_umt5_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    torch.manual_seed(42)
    _ = RESOLUTIONS[resolution]  # resolution is a no-op for UMT5 shapes
    wrapper = UMT5Wrapper(load_umt5())

    vocab_size = wrapper.encoder.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 512), dtype=torch.long)
    attention_mask = torch.ones(1, 512, dtype=torch.long)

    # NOTE: sharding functions operate on sub-module paths, but our wrapper
    # exposes the encoder at self.encoder — pass a closure that unwraps.
    shard_fn = None
    if sharded:

        def shard_fn(model, mesh):
            shard_umt5_weights(model.encoder, mesh)

    compare_cpu_tt(wrapper, [input_ids, attention_mask], shard_fn=shard_fn)
