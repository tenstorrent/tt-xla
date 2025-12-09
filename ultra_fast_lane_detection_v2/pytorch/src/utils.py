# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Ported from https://github.com/tenstorrent/tt-metal/blob/main/models/demos/ufld_v2/reference/ufld_v2_model.py

"""
Utility functions for Ultra-Fast-Lane-Detection-v2 model
"""

import torch


def load_model(model_path, input_height=320, input_width=800):
    """Load a TuSimple34 model with pretrained weights

    Args:
        model_path: Path to the pretrained model weights (.pth file)
        input_height: Input height for the model. Default: 320
        input_width: Input width for the model. Default: 800

    Returns:
        torch.nn.Module: Loaded TuSimple34 model
    """
    from .model import TuSimple34

    torch_model = TuSimple34(input_height=input_height, input_width=input_width)
    torch_model.eval()

    # Load weights
    state_dict = torch.load(model_path, map_location="cpu")
    new_state_dict = {}
    for key, value in state_dict["model"].items():
        new_key = key.replace("model.", "res_model.")
        new_state_dict[new_key] = value

    torch_model.load_state_dict(new_state_dict)
    return torch_model
