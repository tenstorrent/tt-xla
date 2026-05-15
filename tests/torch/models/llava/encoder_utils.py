# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""LLaVA CLIPVisionTransformer op tests: exact modules, pixel_values → encoder."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from tests.benchmark.utils import compute_pcc

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
VISION_PIXEL_VALUES_PATH = FIXTURES_DIR / "vision_pixel_values_bf16.pt"
ENCODER_PRE_INPUT_PATH = FIXTURES_DIR / "encoder_pre_input_bf16.pt"
RANDN_SEED = 0x4C1A
RANDN_SEED_ENCODER = 0x4C2B  # different seed from pixel_values randn

# num_stages: 1=embeddings, 2=pre_layernorm, 3..26=+encoder layers 1..24
NUM_VISION_STACK_STAGES = tuple(range(1, 27))

# Encoder-only stack depths (first N layers of CLIPEncoder)
NUM_ENCODER_ONLY_DEPTHS = tuple(range(1, 25))


def vision_stack_stage_name(num_stages: int) -> str:
    if num_stages == 1:
        return "embeddings"
    if num_stages == 2:
        return "pre_layernorm"
    return f"encoder_layer_{num_stages - 2}"


def encoder_only_depth_name(num_layers: int) -> str:
    return f"encoder_depth_{num_layers}"


class CLIPEncoderStackOp(nn.Module):
    """First N layers from loaded vision_model.encoder.layers (exact Module objects)."""

    def __init__(self, layers: nn.ModuleList, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1 or num_layers > len(layers):
            raise ValueError(f"num_layers must be in [1, {len(layers)}], got {num_layers}")
        self.num_layers = num_layers
        for i in range(num_layers):
            self.add_module(f"layer_{i}", layers[i])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            hidden_states = getattr(self, f"layer_{i}")(hidden_states, attention_mask=None)
        return hidden_states


class CLIPVisionThroughEncoderOp(nn.Module):
    """
    Exact CLIPVisionTransformer path from pixel_values, stopping at:

    1  -> CLIPVisionEmbeddings (1a+1b+1c)
    2  -> + pre_layernorm
    3  -> + encoder layer 0
    26 -> + all 24 encoder layers
    """

    def __init__(
        self,
        embeddings: nn.Module,
        pre_layernorm: nn.Module,
        encoder_layers: nn.ModuleList,
        num_stages: int,
    ) -> None:
        super().__init__()
        if num_stages < 1 or num_stages > 26:
            raise ValueError(f"num_stages must be in [1, 26], got {num_stages}")
        self.embeddings = embeddings
        self.pre_layernorm = pre_layernorm
        self.num_stages = num_stages
        num_encoder_layers = max(0, num_stages - 2)
        for i in range(num_encoder_layers):
            self.add_module(f"encoder_layer_{i}", encoder_layers[i])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=False)
        if self.num_stages == 1:
            return hidden_states
        hidden_states = self.pre_layernorm(hidden_states)
        if self.num_stages == 2:
            return hidden_states
        for i in range(self.num_stages - 2):
            hidden_states = getattr(self, f"encoder_layer_{i}")(
                hidden_states, attention_mask=None
            )
        return hidden_states


def load_vision_pixel_values(path: Path | None = None) -> torch.Tensor:
    path = path or VISION_PIXEL_VALUES_PATH
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["pixel_values"]


def load_encoder_pre_input(path: Path | None = None) -> torch.Tensor:
    path = path or ENCODER_PRE_INPUT_PATH
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["hidden_states"]


def make_randns_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(reference.shape, dtype=reference.dtype, generator=generator)


def make_randn_pixel_values(reference: torch.Tensor, seed: int = RANDN_SEED) -> torch.Tensor:
    return make_randns_like(reference, seed)


def tensor_pcc(device_output: torch.Tensor, golden_output: torch.Tensor) -> float:
    device = device_output.detach().cpu().float()
    golden = golden_output.detach().cpu().float()
    return compute_pcc(golden, device)
