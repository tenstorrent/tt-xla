# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import MochiTransformer3DModel

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


def run_transformer():
    """
    Test Mochi DiT (Diffusion Transformer) with minimal config.

    Uses random weights and reduced layers for fast runtime testing.
    Activation shapes are identical to full model (layers don't change dimensions).

    Input:
    - hidden_states: Noisy latents [B, 12, t, h, w]
    - timestep: Diffusion step index [B] (LongTensor, range 0-999)
    - text_encoder_hidden_states: Text embeddings [B, seq_len, 4096] from T5-XXL
    - text_encoder_attention_mask: Text attention mask [B, seq_len]

    Output: Transformer2DModelOutput with .sample [B, 12, t, h, w]

    Minimal config (vs full Mochi):
    - 2 layers (vs 48)
    - Same attention: 24 heads, 128 dim each
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Create minimal transformer with random weights (no pretrained loading)
    # Uses 2 layers instead of 48, same architecture otherwise
    transformer = MochiTransformer3DModel(
        num_layers=4,  # 48 → 4 for fast testing
    ).to(torch.bfloat16)

    # Prepare inputs
    # hidden_states (noisy latents): [B, C, T, H, W] = [1, 12, 2, 8, 8]
    hidden_states = torch.randn(1, 12, 2, 60, 106, dtype=torch.bfloat16)

    # timestep: diffusion step index as LongTensor [B]
    # Mochi uses 1000 diffusion steps, so valid range is 0-999
    timestep = torch.tensor([500], dtype=torch.long)

    text_seq_len = 128
    # text_encoder_hidden_states (text embeddings from T5-XXL): [B, text_seq_len, 4096]
    # Using minimal 16 tokens for runtime testing
    text_encoder_hidden_states = torch.randn(
        1, text_seq_len, 4096, dtype=torch.bfloat16
    )

    # text_encoder_attention_mask: [B, text_seq_len]
    # All tokens valid (no padding)
    text_encoder_attention_mask = torch.ones(1, text_seq_len, dtype=torch.long)

    # Compile with TT backend
    transformer = transformer.eval().to(device)
    transformer = torch.compile(transformer, backend="tt")

    # Move inputs to device
    hidden_states = hidden_states.to(device)
    timestep = timestep.to(device)
    text_encoder_hidden_states = text_encoder_hidden_states.to(device)
    text_encoder_attention_mask = text_encoder_attention_mask.to(device)

    with torch.no_grad():
        # DiT forward pass
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=text_encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=text_encoder_attention_mask,
        )

    print(
        f"Transformer output shape: {output.sample.shape}, expected shape: {list(hidden_states.shape)} (same as input)"
    )


if __name__ == "__main__":
    print("Running Mochi DiT (minimal) test...")
    run_transformer()
