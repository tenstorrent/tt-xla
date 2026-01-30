# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.runtime as xr
from diffusers import MochiTransformer3DModel
from infra import Framework, run_graph_test
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_mochi_dit():
    """
    Test Mochi DiT (Diffusion Transformer) with minimal config.

    Uses random weights and reduced layers (4 instead of 48) for fast testing.
    Activation shapes are identical to full model.

    Input shapes:
    - hidden_states: [B, 12, T, H, W] - Noisy latents
    - encoder_hidden_states: [B, seq_len, 4096] - T5-XXL text embeddings
    - timestep: [B] - Diffusion step index (0-999)
    - encoder_attention_mask: [B, seq_len] - Text attention mask

    Output shape: [B, 12, T, H, W] (same as hidden_states)
    """
    xr.set_device_type("TT")

    class Wrapper(torch.nn.Module):
        """
        Wrapper for MochiTransformer3DModel that returns just the sample tensor.

        MochiTransformer3DModel returns Transformer2DModelOutput with .sample property,
        but for testing we need a plain tensor output.
        """

        def __init__(self):
            super().__init__()
            self.transformer = MochiTransformer3DModel(num_layers=4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            output = self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
            )
            return output.sample

    wrapper = Wrapper().to(torch.bfloat16)

    # Input tensors
    hidden_states = torch.randn(1, 12, 2, 60, 106, dtype=torch.bfloat16)
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 128, 4096, dtype=torch.bfloat16)
    encoder_attention_mask = torch.ones(1, 128, dtype=torch.long)

    run_graph_test(
        wrapper,
        [hidden_states, encoder_hidden_states, timestep, encoder_attention_mask],
        framework=Framework.TORCH,
    )
