# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Qwen-Image-2512-Fun-Controlnet-Union model loading and processing.
"""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


REPO_ID = "alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union"


def download_controlnet_weights(filename):
    """Download ControlNet safetensors weights from HuggingFace.

    Args:
        filename: Name of the safetensors file to download.

    Returns:
        str: Local path to the downloaded file.
    """
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


def load_controlnet_state_dict(filename):
    """Load ControlNet weights as a state dict from a safetensors file.

    Args:
        filename: Name of the safetensors file to load.

    Returns:
        dict: The model state dict.
    """
    local_path = download_controlnet_weights(filename)
    return load_file(local_path)


def create_dummy_control_image(height=512, width=512):
    """Create a dummy control conditioning image tensor.

    Simulates a control conditioning image for ControlNet inference.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        torch.Tensor: A dummy control image tensor of shape (1, 3, height, width).
    """
    return torch.zeros(1, 3, height, width, dtype=torch.float32)
