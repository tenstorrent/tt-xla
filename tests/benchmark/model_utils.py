# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for constructing model loaders and handling their inputs, outputs,
and artifacts (CPU transfer, pooling, image saving).

Pure and device-free (operates on already-produced tensors / loader classes).
"""

import inspect
from typing import Optional

import torch
from PIL import Image


def create_model_loader(ModelLoader, num_layers: Optional[int] = None, *args, **kwargs):
    """Create a model loader with optional num_layers override.

    Returns None if num_layers is requested but the loader does not support it.
    """
    if num_layers is None:
        return ModelLoader(*args, **kwargs)
    params = inspect.signature(ModelLoader.__init__).parameters
    supports_num_layers = "num_layers" in params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    if not supports_num_layers:
        return None
    return ModelLoader(*args, num_layers=num_layers, **kwargs)


def apply_mean_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool token embeddings [B, S, H] over masked positions -> [B, H]."""
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    )
    sentence_embeddings = torch.sum(
        hidden_states * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sentence_embeddings


def apply_last_token_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pool the last non-padding token of each sequence ([B, S, H] -> [B, H])."""
    # Left padding => last column is the real final token for every sequence.
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
    if left_padding:
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[
        torch.arange(batch_size, device=hidden_states.device), sequence_lengths
    ]


def move_to_cpu(data):
    """Recursively move all tensors in a data structure to CPU.

    Handles dicts, lists, tuples, and HuggingFace ModelOutput objects.
    Preserves the original data structure types.
    """
    if isinstance(data, torch.Tensor):
        return data.cpu()
    # HF ModelOutput (an OrderedDict subclass) must be checked before plain dict
    # and mutated in place to preserve its type.
    # to_tuple() is what distinguishes it from a plain dict.
    elif hasattr(data, "to_tuple") and hasattr(data, "keys"):
        for key in list(data.keys()):
            value = data[key]
            if isinstance(value, torch.Tensor):
                data[key] = value.cpu()
            elif value is not None:
                data[key] = move_to_cpu(value)
        return data
    elif isinstance(data, dict):
        return {k: move_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        moved = [move_to_cpu(item) for item in data]
        return type(data)(moved)
    return data


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    """Save a diffusion-model output tensor (range [-1, 1], CHW or BCHW) as a PNG."""
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)
