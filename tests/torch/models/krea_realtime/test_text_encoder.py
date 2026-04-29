# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Krea Realtime — UMT5 text encoder component test.

IN:  input_ids       (1, 512) int64
     attention_mask  (1, 512) int64
OUT: last_hidden_state (1, 512, 4096) bfloat16
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.torch.models.krea_realtime.shared import (
    MAX_SEQ_LEN,
    krea_mesh,
    load_text_encoder,
    shard_text_encoder_specs,
)


@pytest.mark.skip(
    reason="OOM on single device — UMT5-XXL exceeds single-chip memory; sharded variant runs"
)
def test_text_encoder():
    _run(sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
def test_text_encoder_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    encoder = load_text_encoder()

    vocab_size = encoder.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, MAX_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    mesh = krea_mesh() if sharded else None
    shard_spec_fn = shard_text_encoder_specs if sharded else None

    run_graph_test(
        encoder,
        [input_ids, attention_mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
