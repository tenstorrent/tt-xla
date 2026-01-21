# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility wrappers for Mochi models."""

import torch


class MochiVAEWrapper(torch.nn.Module):
    """
    Wrapper for Mochi VAE that handles tiled vs non-tiled execution paths.

    The wrapper unifies the forward() interface between:
    - Tiled mode: Uses vae.decode() which returns object with .sample attribute
    - Non-tiled mode: Uses direct decoder forward() which returns tensor

    This allows the test infrastructure to treat both variants uniformly.
    """

    def __init__(self, vae_model: torch.nn.Module, enable_tiling: bool):
        """
        Initialize wrapper.

        Args:
            vae_model: Either full VAE model (tiled) or decoder module (non-tiled)
            enable_tiling: Whether tiling is enabled
        """
        super().__init__()
        self.model = vae_model
        self.enable_tiling = enable_tiling

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VAE decoder.

        Args:
            latent: Normalized latent tensor [B, 12, t, h, w]

        Returns:
            Decoded video tensor [B, 3, T, H, W]
        """
        if self.enable_tiling:
            # Tiled mode: vae.decode() returns object with .sample attribute
            output = self.model.decode(latent)
            if hasattr(output, "sample"):
                return output.sample
            return output
        else:
            # Non-tiled mode: direct decoder forward pass returns tuple (output, conv_cache)
            output = self.model(latent)
            if isinstance(output, tuple):
                return output[0]
            return output


def calculate_expected_output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Calculate expected output shape for Mochi VAE decoder.
    Temporal expansion: 6
    Spatial expansion: 8x8

    Args:
        input_shape: Shape of input tensor [B, 12, t, h, w]

    Returns:
        Shape of output tensor [B, 3, T, H, W]
    """
    return (
        input_shape[0],
        3,
        input_shape[2] * 6,
        input_shape[3] * 8,
        input_shape[4] * 8,
    )
