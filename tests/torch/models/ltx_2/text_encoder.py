# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


def run_text_encoder():
    """
    Test Gemma3 text encoder in isolation with minimal config.

    Uses random weights with 2 layers (vs 48 in real Gemma3-12B).
    hidden_size=3840 matches LTX-2's caption_channels for connector compatibility.

    The real model extracts features from ALL decoder layers (output_hidden_states=True),
    which are then processed by LTX2TextConnectors (separate test).

    Minimal config (vs real Gemma3-12B):
    - 2 layers (vs 48)
    - Same hidden_size: 3840 (matches caption_channels)

    Input: Token IDs [B, seq_len]
    Output: Hidden states from all layers [num_layers x [B, seq_len, 3840]]
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Create minimal Gemma3 with random weights
    # hidden_size=3840 must match LTX-2 caption_channels
    # num_hidden_layers=2 gives 3 hidden states (2 layers + 1 embedding)
    text_config = Gemma3TextConfig(
        hidden_size=3840,
        intermediate_size=15360,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        vocab_size=262208,
    )
    config = Gemma3Config(text_config=text_config, vision_config=None)
    text_encoder = Gemma3ForConditionalGeneration(config).to(torch.bfloat16)

    text_encoder = text_encoder.eval().to(device)
    text_encoder = torch.compile(text_encoder, backend="tt")

    # Use random token IDs (no tokenizer needed with random weights)
    seq_len = 128
    input_ids = torch.randint(0, 262208, (1, seq_len), dtype=torch.long).to(device)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    torch_xla.sync()

    num_layers = len(output.hidden_states)
    last_shape = output.hidden_states[-1].shape
    print(f"Number of hidden state layers: {num_layers}")
    print(f"Last hidden state shape: {last_shape}")
    print(f"Expected: {num_layers} layers, each [1, {seq_len}, 3840]")


if __name__ == "__main__":
    print("Running LTX-2 Text Encoder (minimal 2-layer Gemma3) test...")
    run_text_encoder()
