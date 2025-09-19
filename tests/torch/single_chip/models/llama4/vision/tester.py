# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence, Tuple
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama4.modeling_llama4 import Llama4VisionModel


def reshape_for_broadcast(freqs: torch.Tensor, query: torch.Tensor):
    """Helper function to reshape frequency tensors for broadcasting"""
    ndim = query.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(query.shape)]
    return freqs.view(*shape)


def real_valued_vision_apply_rotary_emb(
    query: torch.Tensor, key: torch.Tensor, freqs_ci: torch.Tensor = None, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embedding using real-valued arithmetic instead of complex numbers.

    This implements the same rotation as complex multiplication but using only real tensors:
    For a complex number z = x + iy rotated by angle θ:
    z' = z * e^(iθ) = z * (cos(θ) + i*sin(θ))
    Real part: x' = x*cos(θ) - y*sin(θ)
    Imag part: y' = x*sin(θ) + y*cos(θ)
    """
    if freqs_ci is None:
        return query, key

    # Handle case where freqs_ci is a tuple (cos, sin)
    if isinstance(freqs_ci, tuple):
        cos_vals, sin_vals = freqs_ci
        cos_vals = cos_vals.to(query.device)
        sin_vals = sin_vals.to(query.device)
    else:
        # Extract cos and sin from the combined frequency tensor
        # freqs_ci is typically structured as [cos_values, sin_values] or interleaved
        # For vision models, freqs_ci often contains both cos and sin components

        device = query.device

        # Use a simpler approach that doesn't involve dynamic slicing
        # Just use the freqs_ci tensor directly and broadcast it properly
        if freqs_ci.dim() >= 2:
            # Take the appropriate slice based on the expected dimensions
            freq_slice = freqs_ci[..., 0 : query.size(-1) // 2]
        else:
            # If freqs_ci is 1D, expand it to match query dimensions
            freq_slice = freqs_ci[: query.size(-1) // 2]

        cos_vals = torch.cos(freq_slice).to(device)
        sin_vals = torch.sin(freq_slice).to(device)

    # Split query and key into pairs (treating adjacent dims as real/imaginary pairs)
    # Shape: [..., head_dim] -> [..., head_dim//2, 2]
    query_reshaped = query.float().view(*query.shape[:-1], -1, 2)
    key_reshaped = key.float().view(*key.shape[:-1], -1, 2)

    # Extract real and imaginary parts
    query_real = query_reshaped[..., 0]  # [..., head_dim//2]
    query_imag = query_reshaped[..., 1]  # [..., head_dim//2]
    key_real = key_reshaped[..., 0]  # [..., head_dim//2]
    key_imag = key_reshaped[..., 1]  # [..., head_dim//2]

    # Ensure cos/sin have compatible shapes for broadcasting
    cos_vals = reshape_for_broadcast(cos_vals, query_real)
    sin_vals = reshape_for_broadcast(sin_vals, query_real)

    # Apply rotation using real arithmetic
    # Real part: x' = x*cos - y*sin
    # Imag part: y' = x*sin + y*cos
    query_out_real = query_real * cos_vals - query_imag * sin_vals
    query_out_imag = query_real * sin_vals + query_imag * cos_vals
    key_out_real = key_real * cos_vals - key_imag * sin_vals
    key_out_imag = key_real * sin_vals + key_imag * cos_vals

    # Recombine real and imaginary parts
    query_out = torch.stack([query_out_real, query_out_imag], dim=-1)
    key_out = torch.stack([key_out_real, key_out_imag], dim=-1)

    # Reshape back to original shape
    query_out = query_out.view(*query.shape).type_as(query)
    key_out = key_out.view(*key.shape).type_as(key)

    return query_out, key_out


# Real-valued rotary embedding to avoid complex tensors
class RealValuedVisionRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        idx = config.image_size // config.patch_size
        img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # ID_CLS_TOKEN
        frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
        frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
        freq_dim = config.hidden_size // config.num_attention_heads // 2
        rope_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, freq_dim, 2)[: (freq_dim // 2)].float() / freq_dim)
        )
        freqs_x = (
            (frequencies_x + 1)[..., None] * rope_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs_y = (
            (frequencies_y + 1)[..., None] * rope_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)

        # Store cos and sin separately instead of as complex numbers
        self.freqs_cos = torch.cos(freqs)
        self.freqs_sin = torch.sin(freqs)

    def forward(self, pixel_values):
        return self.freqs_cos.to(pixel_values.device), self.freqs_sin.to(
            pixel_values.device
        )


# Temporary monkeypatch to avoid complex tensors, need to revisit this
class MockVisionRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        self.num_patches = (image_size // patch_size) ** 2 + 1
        self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(self, pixel_values):
        # CRITICAL: Create tensor with float32 dtype to avoid XLA issues
        return torch.ones(
            self.num_patches,
            self.num_patches,
            self.head_dim,
            device=pixel_values.device,
            dtype=torch.float32,
        )



class Llama4VisionTester(TorchModelTester):
    """Tester for Llama4Vision model."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        model_name = "meta-llama/Llama-4-Scout-17B-16E"

        # Get the full config but only use vision part
        full_config = AutoConfig.from_pretrained(model_name)
        original_vision_config = full_config.vision_config

        # Create a NEW vision config with smaller dimensions
        from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig

        vision_config = Llama4VisionConfig(
            hidden_size=64,
            intermediate_size=16,  # After pixel shuffle: 64/(2²) = 16
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=64,
            patch_size=16,
            num_channels=3,
            attention_dropout=0.0,
            rope_theta=getattr(original_vision_config, "rope_theta", 10000.0),
            vision_output_dim=64,
            projector_input_dim=32,  # fc1: 16 → 32
            projector_output_dim=32,  # fc2: 32 → 32
            pixel_shuffle_ratio=2,
            projector_dropout=0.0,
        )

        # Create vision-only model
        self.model = Llama4VisionModel(vision_config)
        # self.model = self.model.to(torch.bfloat16)
        self.model.eval()

        # Replace vision rotary embedding with real-valued version
        self.model.rotary_embedding = RealValuedVisionRotaryEmbedding(vision_config)

        # CRITICAL: Also replace the vision_apply_rotary_emb function
        import transformers.models.llama4.modeling_llama4 as llama4_mod

        # Monkey patch the module-level function to use real-valued version
        llama4_mod.vision_apply_rotary_emb = real_valued_vision_apply_rotary_emb

        return self.model

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        # Create dummy image inputs
        # Standard vision input: pixel values with shape [batch, channels, height, width]
        vision_config = self.model.config

        batch_size = 1
        channels = vision_config.num_channels  # Usually 3 (RGB)
        height = vision_config.image_size  # e.g., 448
        width = vision_config.image_size

        pixel_values = torch.randn(
            batch_size, channels, height, width, dtype=torch.bfloat16
        )

        # CRITICAL: Ensure ALL tensors are float type and on same device
        # This prevents "CPULongType" XLA tensor errors
        pixel_values = pixel_values.float()  # Convert to float32 for XLA compatibility

        inputs = {
            "pixel_values": pixel_values,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        return inputs
