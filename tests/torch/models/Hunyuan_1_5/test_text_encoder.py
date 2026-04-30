# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HunyuanVideo 1.5 — Qwen2.5-VL text encoder component test.

Size: 7.07B params

IN:  input_ids       (1, 1108) int64
     attention_mask  (1, 1108) int64
OUT: model output (Qwen2_5_VLTextModelOutputWithPast) — last_hidden_state shape (1, 1108, 3584) bfloat16
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.torch.models.Hunyuan_1_5.shared import (
    TEXT_TOKEN_MAX_LEN,
    hunyuan_mesh,
    load_text_encoder,
    shard_text_encoder_specs,
)


@pytest.mark.skip(
    reason="model size > 7B — won't fit on a single chip; sharded variant runs"
)
def test_text_encoder():
    _run(sharded=False)


@pytest.mark.xfail(
    reason="PCC drop (got 0.9712937232386395, required >= 0.99) — https://github.com/tenstorrent/tt-xla/issues/4484"
)
@pytest.mark.nightly
@pytest.mark.model_test
def test_text_encoder_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    encoder = load_text_encoder()

    vocab_size = encoder.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, TEXT_TOKEN_MAX_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, TEXT_TOKEN_MAX_LEN, dtype=torch.long)

    mesh = hunyuan_mesh() if sharded else None
    shard_spec_fn = shard_text_encoder_specs if sharded else None

    run_graph_test(
        encoder,
        [input_ids, attention_mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
