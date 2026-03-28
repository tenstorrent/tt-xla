# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Wan Fun Control model loading."""

import torch


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

    The model stores config.json and diffusion_pytorch_model.safetensors at the
    repo root (no subfolder), so we load directly from the model ID.

    Args:
        pretrained_model_name: HuggingFace model ID
        dtype: Torch dtype for model weights
    """
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        pretrained_model_name,
        torch_dtype=dtype,
    )
    transformer.eval()
    return transformer


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
