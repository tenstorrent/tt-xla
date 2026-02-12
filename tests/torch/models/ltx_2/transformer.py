# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import LTX2VideoTransformer3DModel

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


def run_transformer():
    """
    Test LTX-2 Dual-Stream Transformer with minimal config.

    Uses random weights and 4 layers (vs 48 in full model) for fast testing.
    Tests the full dual-stream path: video + audio tokens processed together.

    The LTX-2 transformer expects PRE-PACKED token sequences [B, seq_len, dim],
    unlike Mochi which takes 5D [B, C, T, H, W] and patchifies internally.

    Input:
    - hidden_states: Video tokens [B, N_v, 128]
    - audio_hidden_states: Audio tokens [B, N_a, 128]
    - encoder_hidden_states: Text embeddings for video [B, L, 3840]
    - audio_encoder_hidden_states: Text embeddings for audio [B, L, 3840]
    - timestep: Diffusion step index [B] (LongTensor)
    - encoder_attention_mask: Text mask [B, L]
    - audio_encoder_attention_mask: Audio text mask [B, L]

    Output:
    - .sample: Denoised video tokens [B, N_v, 128]
    - .audio_sample: Denoised audio tokens [B, N_a, 128]

    Minimal config (vs full LTX-2):
    - 4 layers (vs 48)
    - Same dimensions: video 32 heads x 128 dim, audio 32 heads x 64 dim
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Create minimal transformer with random weights
    transformer = LTX2VideoTransformer3DModel(
        num_layers=4,
    ).to(torch.bfloat16)

    # Video tokens: [B, N_v, 128] where N_v = num_frames * height * width
    # 2 frames, 4x4 spatial -> 32 tokens
    num_frames, h, w = 2, 4, 4
    n_video = num_frames * h * w  # 32
    hidden_states = torch.randn(1, n_video, 128, dtype=torch.bfloat16)

    # Audio tokens: [B, N_a, 128]
    n_audio = 16
    audio_hidden_states = torch.randn(1, n_audio, 128, dtype=torch.bfloat16)

    # Text embeddings: [B, L, caption_channels=3840]
    text_seq_len = 16
    encoder_hidden_states = torch.randn(1, text_seq_len, 3840, dtype=torch.bfloat16)
    audio_encoder_hidden_states = torch.randn(
        1, text_seq_len, 3840, dtype=torch.bfloat16
    )

    # Timestep and attention masks
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_attention_mask = torch.ones(1, text_seq_len, dtype=torch.long)
    audio_encoder_attention_mask = torch.ones(1, text_seq_len, dtype=torch.long)

    # Compile with TT backend
    transformer = transformer.eval().to(device)
    transformer = torch.compile(transformer, backend="tt")

    # Move inputs to device
    hidden_states = hidden_states.to(device)
    audio_hidden_states = audio_hidden_states.to(device)
    encoder_hidden_states = encoder_hidden_states.to(device)
    audio_encoder_hidden_states = audio_encoder_hidden_states.to(device)
    timestep = timestep.to(device)
    encoder_attention_mask = encoder_attention_mask.to(device)
    audio_encoder_attention_mask = audio_encoder_attention_mask.to(device)

    with torch.no_grad():
        output = transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=num_frames,
            height=h,
            width=w,
        )

    torch_xla.sync()
    print(
        f"Video output shape: {output.sample.shape}, " f"expected: [1, {n_video}, 128]"
    )
    print(
        f"Audio output shape: {output.audio_sample.shape}, "
        f"expected: [1, {n_audio}, 128]"
    )


def load_model():
    transformer = LTX2VideoTransformer3DModel(
        num_layers=4,
    ).to(torch.bfloat16)
    return transformer


def load_inputs():
    # Video tokens: [B, N_v, 128] where N_v = num_frames * height * width
    # 2 frames, 4x4 spatial -> 32 tokens
    num_frames, h, w = 2, 4, 4
    n_video = num_frames * h * w  # 32
    hidden_states = torch.randn(1, n_video, 128, dtype=torch.bfloat16)

    # Audio tokens: [B, N_a, 128]
    n_audio = 16
    audio_hidden_states = torch.randn(1, n_audio, 128, dtype=torch.bfloat16)

    # Text embeddings: [B, L, caption_channels=3840]
    text_seq_len = 16
    encoder_hidden_states = torch.randn(1, text_seq_len, 3840, dtype=torch.bfloat16)
    audio_encoder_hidden_states = torch.randn(
        1, text_seq_len, 3840, dtype=torch.bfloat16
    )

    # Timestep and attention masks
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_attention_mask = torch.ones(1, text_seq_len, dtype=torch.long)
    audio_encoder_attention_mask = torch.ones(1, text_seq_len, dtype=torch.long)

    # make dict of inputs
    inputs = {
        "hidden_states": hidden_states,
        "audio_hidden_states": audio_hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "audio_encoder_hidden_states": audio_encoder_hidden_states,
        "timestep": timestep,
        "encoder_attention_mask": encoder_attention_mask,
        "audio_encoder_attention_mask": audio_encoder_attention_mask,
        "num_frames": num_frames,
        "height": h,
        "width": w,
    }

    return inputs


if __name__ == "__main__":
    print("Running LTX-2 Transformer (4-layer, random weights) test...")
    run_transformer()
