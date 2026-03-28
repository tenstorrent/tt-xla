# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for ControlNet-LLLite model loading and processing.

ControlNet-LLLite is a lightweight ControlNet variant for SDXL that adds small
control modules to the UNet's attention layers. The modules are stored as
individual safetensors files.
"""

import re
from collections import OrderedDict

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class LLLiteModule(nn.Module):
    """A single LLLite control module consisting of down/mid/up projections."""

    def __init__(self, down_weight, mid_weight, up_weight):
        super().__init__()
        self.down = nn.Linear(down_weight.shape[1], down_weight.shape[0], bias=False)
        self.mid = nn.Linear(mid_weight.shape[1], mid_weight.shape[0], bias=False)
        self.up = nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=False)

        self.down.weight = nn.Parameter(down_weight)
        self.mid.weight = nn.Parameter(mid_weight)
        self.up.weight = nn.Parameter(up_weight)

    def forward(self, x):
        x = self.down(x)
        x = torch.nn.functional.silu(x)
        x = self.mid(x)
        x = torch.nn.functional.silu(x)
        x = self.up(x)
        return x


class ControlNetLLLite(nn.Module):
    """ControlNet-LLLite model assembled from safetensors state dict.

    Groups the loaded weights into individual LLLite modules by parsing
    the state dict key naming convention.
    """

    def __init__(self, state_dict):
        super().__init__()
        self.modules_dict = nn.ModuleDict()

        # Group keys by module name (everything before .down/.mid/.up)
        module_groups = OrderedDict()
        for key in sorted(state_dict.keys()):
            match = re.match(r"(.+)\.(down|mid|up)\.weight", key)
            if match:
                module_name = match.group(1).replace(".", "_")
                if module_name not in module_groups:
                    module_groups[module_name] = {}
                module_groups[module_name][match.group(2)] = state_dict[key]

        for name, weights in module_groups.items():
            if "down" in weights and "mid" in weights and "up" in weights:
                self.modules_dict[name] = LLLiteModule(
                    weights["down"], weights["mid"], weights["up"]
                )

    def forward(self, x):
        """Forward pass through all LLLite modules, summing their outputs."""
        out = torch.zeros_like(x)
        for module in self.modules_dict.values():
            out = out + module(x)
        return out


def load_controlnet_lllite(repo_id, filename):
    """Download and load a ControlNet-LLLite model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g. "bdsqlsz/qinglong_controlnet-lllite")
        filename: Safetensors filename to download

    Returns:
        ControlNetLLLite: The loaded model in eval mode
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = load_file(model_path)

    model = ControlNetLLLite(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def create_dummy_input(model, batch_size=1):
    """Create a dummy input tensor for the ControlNet-LLLite model.

    Infers the input dimension from the first module's down projection layer.

    Args:
        model: ControlNetLLLite model instance
        batch_size: Batch size for the input tensor

    Returns:
        torch.Tensor: Dummy input tensor of shape [batch_size, input_dim]
    """
    first_module = next(iter(model.modules_dict.values()))
    input_dim = first_module.down.weight.shape[1]

    torch.manual_seed(42)
    return torch.randn(batch_size, input_dim)
