# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CogVideoX-5b — T5 text encoder component test.

IN:  input_ids       (1, 226) int64
     attention_mask  (1, 226) int64
OUT: last_hidden_state (1, 226, 4096) bfloat16
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.torch.models.cogvideox_5b.shared import (
    MAX_SEQ_LEN,
    cogvideox_mesh,
    load_text_encoder,
    shard_text_encoder_specs,
)


class T5EncoderWrapper(torch.nn.Module):
    """Return last_hidden_state as a plain tensor (not a model output object)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


def test_text_encoder():
    _run(sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
def test_text_encoder_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    wrapper = T5EncoderWrapper(load_text_encoder()).eval()

    vocab_size = wrapper.encoder.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, MAX_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    mesh = cogvideox_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_text_encoder_specs(m.encoder)) if sharded else None

    run_graph_test(
        wrapper,
        [input_ids, attention_mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
