# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Wan Fun Control model loading."""

import torch


# Wan VAE uses 16 latent channels (z_dim=16)
LATENT_CHANNELS = 16

# Control model has 36 input channels:
# 16 (latent) + 16 (control latent) + 4 (mask/extra conditioning)
TRANSFORMER_IN_CHANNELS = 36

# Small test dimensions
LATENT_HEIGHT = 4
LATENT_WIDTH = 4
LATENT_DEPTH = 2  # temporal latent frames

# Text encoder hidden dim for Wan (umt5-xxl based)
TEXT_HIDDEN_DIM = 4096
TEXT_SEQ_LEN = 8


# ============================================================================
# Model Loading Functions
# ============================================================================


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanTransformer3DModel from the control checkpoint.

    Args:
        pretrained_model_name: HuggingFace model ID
        dtype: Torch dtype for model weights
    """
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer.eval()
    return transformer


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load AutoencoderKLWan from the control checkpoint.

    Args:
        pretrained_model_name: HuggingFace model ID
        dtype: Torch dtype for model weights
    """
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.eval()
    return vae


# ============================================================================
# Input Loading Functions
# ============================================================================


def load_transformer_inputs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Prepare synthetic inputs for WanTransformer3DModel forward pass.

    The control variant uses 36 input channels (16 latent + 20 control/mask).
    """
    batch_size = 1
    seq_len = LATENT_DEPTH * LATENT_HEIGHT * LATENT_WIDTH

    hidden_states = torch.randn(
        batch_size, seq_len, TRANSFORMER_IN_CHANNELS, dtype=dtype
    )
    encoder_hidden_states = torch.randn(
        batch_size, TEXT_SEQ_LEN, TEXT_HIDDEN_DIM, dtype=dtype
    )
    timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "return_dict": False,
    }


def load_vae_decoder_inputs(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load inputs for VAE decoder.

    Returns:
        Latent tensor of shape [1, 16, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH]
    """
    return torch.randn(
        1, LATENT_CHANNELS, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
    )


def load_vae_encoder_inputs(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load inputs for VAE encoder.

    Wan VAE requires frame count T = 1 + 4*N for some integer N.

    Returns:
        RGB video tensor of shape [1, 3, T, H, W]
    """
    num_frames = 1 + 4 * LATENT_DEPTH  # 9 frames
    return torch.randn(
        1, 3, num_frames, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype
    )
