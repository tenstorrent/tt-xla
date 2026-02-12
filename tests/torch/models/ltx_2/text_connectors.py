# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.pipelines.ltx2 import LTX2TextConnectors

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


def run_text_connectors():
    """
    Test LTX2TextConnectors in isolation.

    Uses random weights (no pretrained loading).
    Adapts multi-layer Gemma3 hidden states into separate embeddings
    for the video and audio branches of the transformer.

    Config matches real LTX-2 HuggingFace config except:
    - text_proj_in_factor=3 (vs 49) to match the 2-layer minimal Gemma3
    - num_learnable_registers=None — disables the register-replacement block
      which uses dynamic list comprehensions untraceable by dynamo
    - rope_type="interleaved" — avoids apply_split_rotary_emb which has a
      non-contiguous reshape bug during AOT export tracing

    Input: Concatenated hidden states [B, seq_len, caption_channels * text_proj_in_factor]
           = [B, 256, 3840 * 3] = [B, 256, 11520]
    Output: video_text [B, 256, 3840], audio_text [B, 256, 3840], mask [B, 256]
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    connectors = LTX2TextConnectors(
        caption_channels=3840,
        text_proj_in_factor=3,
        video_connector_num_attention_heads=30,
        video_connector_attention_head_dim=128,
        video_connector_num_layers=2,
        video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=30,
        audio_connector_attention_head_dim=128,
        audio_connector_num_layers=2,
        audio_connector_num_learnable_registers=None,
        connector_rope_base_seq_len=4096,
        rope_theta=10000.0,
        rope_double_precision=True,
        causal_temporal_positioning=False,
        rope_type="interleaved",
    ).to(torch.bfloat16)

    connectors = connectors.eval().to(device)
    connectors = torch.compile(connectors, backend="tt")

    # Input: concatenated hidden states from all Gemma3 layers
    # Shape: [B, seq_len, caption_channels * text_proj_in_factor]
    seq_len = 256
    text_hidden_states = torch.randn(1, seq_len, 3840 * 3, dtype=torch.bfloat16).to(
        device
    )
    # Binary mask: all-ones means all tokens valid, no padding.
    # forward() converts this to additive: (1 - 1) * float_max = 0 (no masking).
    attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

    with torch.no_grad():
        video_text, audio_text, new_mask = connectors(
            text_hidden_states, attention_mask
        )

    print(f"Video text embeddings shape: {video_text.shape}")
    print(f"Audio text embeddings shape: {audio_text.shape}")
    print(f"New attention mask shape: {new_mask.shape}")
    print(f"Expected: [1, {seq_len}, 3840] for each branch, mask [1, {seq_len}]")


def load_model():
    connectors = LTX2TextConnectors(
        caption_channels=3840,
        text_proj_in_factor=3,
        video_connector_num_attention_heads=30,
        video_connector_attention_head_dim=128,
        video_connector_num_layers=2,
        video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=30,
        audio_connector_attention_head_dim=128,
        audio_connector_num_layers=2,
        audio_connector_num_learnable_registers=None,
        connector_rope_base_seq_len=4096,
        rope_theta=10000.0,
        rope_double_precision=True,
        causal_temporal_positioning=False,
        rope_type="interleaved",
    ).to(torch.bfloat16)

    return connectors


def load_inputs():
    seq_len = 256
    text_hidden_states = torch.randn(1, seq_len, 3840 * 3, dtype=torch.bfloat16)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    return text_hidden_states, attention_mask


if __name__ == "__main__":
    print("Running LTX-2 Text Connectors test...")
    run_text_connectors()
