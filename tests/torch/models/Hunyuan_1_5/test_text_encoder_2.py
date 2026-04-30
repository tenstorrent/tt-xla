# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HunyuanVideo 1.5 — ByT5 text encoder component test (text_encoder_2).

Size: 0.22B params

IN:  input_ids       (1, 256) int64
     attention_mask  (1, 256) float32
OUT: model output (BaseModelOutput) — last_hidden_state shape (1, 256, 1472) bfloat16
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.torch.models.Hunyuan_1_5.shared import (
    TEXT_TOKEN_2_MAX_LEN,
    load_text_encoder_2,
)


@pytest.mark.nightly
@pytest.mark.model_test
def test_text_encoder_2():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    encoder = load_text_encoder_2()

    vocab_size = encoder.config.vocab_size
    input_ids = torch.randint(
        0, vocab_size, (1, TEXT_TOKEN_2_MAX_LEN), dtype=torch.long
    )
    # Pipeline calls text_encoder_2 with attention_mask.float() — match that
    attention_mask = torch.ones(1, TEXT_TOKEN_2_MAX_LEN, dtype=torch.float32)

    run_graph_test(
        encoder,
        [input_ids, attention_mask],
        framework=Framework.TORCH,
    )
