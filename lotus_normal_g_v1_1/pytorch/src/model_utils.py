# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Lotus Normal G v1.1 model loading and preprocessing.
"""

import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def load_unet(pretrained_model_name, dtype=None):
    """Load the UNet2DConditionModel from the Lotus Normal model.

    Args:
        pretrained_model_name: HuggingFace model identifier.
        dtype: Optional torch dtype override.

    Returns:
        UNet2DConditionModel: The loaded UNet model.
    """
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name, subfolder="unet", **model_kwargs
    )
    unet.eval()
    return unet


def prepare_unet_inputs(pretrained_model_name, dtype=None):
    """Prepare synthetic inputs for the Lotus Normal UNet.

    The UNet expects:
    - sample: latent tensor of shape [B, 8, H, W] (8 = 4 RGB latent + 4 prediction latent)
    - timestep: diffusion timestep tensor
    - encoder_hidden_states: text encoder output of shape [B, seq_len, 1024]
    - class_labels: task embedding of shape [B, 4]

    Args:
        pretrained_model_name: HuggingFace model identifier.
        dtype: Optional torch dtype override.

    Returns:
        dict: Dictionary of input tensors for the UNet.
    """
    batch_size = 1
    latent_height = 96
    latent_width = 96

    # Latent input: 8 channels (concatenation of RGB latent + prediction latent)
    sample = torch.randn(batch_size, 8, latent_height, latent_width)

    # Single-step diffusion at timestep 999
    timestep = torch.tensor([999])

    # Text encoder hidden states (empty prompt encoded)
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name, subfolder="text_encoder"
    )
    text_encoder.eval()

    text_inputs = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]

    # Task embedding for normal estimation: sin/cos encoding of [1, 0]
    task_emb = torch.tensor([[1.0, 0.0]])
    class_labels = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    if dtype is not None:
        sample = sample.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)
        class_labels = class_labels.to(dtype)

    return {
        "sample": sample,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "class_labels": class_labels,
    }
