# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Text Encoder (Gemma 3) — standalone bringup script.

Component: Gemma3TextModel (inner language model, avoids conditional generation wrapper)

Uses minimal 2-layer config with random weights for initial bringup.
hidden_size=3840 matches LTX-2 caption_channels for connector compatibility.

Known issues with full Gemma3ForConditionalGeneration:
  - slice(-N) with N > 127 overflows int8 in XLA HLO (from causal mask generation)
  - SPMD sharding + graph breaks cause device count mismatch (4 vs 1)

Workaround: Use Gemma3TextModel directly with pre-computed attention mask
to avoid dynamic mask generation with large negative slice indices.

Input:  input_ids [B, seq_len], attention_mask [B, 1, seq_len, seq_len]
Output: hidden_states from all layers
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel


def run_ltx2_text_encoder():
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Minimal 2-layer Gemma3 config matching LTX-2 dimensions.
    text_config = Gemma3TextConfig(
        hidden_size=3840,
        intermediate_size=15360,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        vocab_size=262208,
        # Disable sliding window to avoid slice overflow
        sliding_window=None,
    )

    model = Gemma3TextModel(text_config).to(torch.bfloat16)
    model.config.use_cache = False
    model = model.eval()

    model = model.to(device)
    model = torch.compile(model, backend="tt")

    # Random token IDs
    seq_len = 128
    input_ids = torch.randint(0, 262208, (1, seq_len), dtype=torch.long).to(device)

    # Pre-compute causal attention mask on CPU then move to device.
    # This avoids dynamic mask generation with large negative slice indices.
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16), diagonal=1
    ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    causal_mask = causal_mask.to(device)

    # Warm-up pass (compilation)
    print("Text Encoder (Gemma 3, 2-layer minimal): warm-up pass (compilation)...")
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            output_hidden_states=True,
        )
    torch_xla.sync(wait=True)
    hidden_states = output.hidden_states
    print(f"  Number of hidden states: {len(hidden_states)}")
    print(f"  Each hidden state shape: {hidden_states[0].shape}")

    # Timed pass
    print("Text Encoder (Gemma 3, 2-layer minimal): timed pass...")
    start = time.time()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            output_hidden_states=True,
        )
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    hidden_states = output.hidden_states
    print(f"  Number of hidden states: {len(hidden_states)}")
    print(f"  Inference time: {elapsed:.3f}s")

    return hidden_states


if __name__ == "__main__":
    run_ltx2_text_encoder()
