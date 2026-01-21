# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

# Channel-wise standard deviations for VAE latent normalization
# Source: Mochi VAE implementation (12 latent channels)
VAE_STD_CHANNELS = [
    0.925,
    0.934,
    0.946,
    0.939,
    0.961,
    1.033,
    0.979,
    1.024,
    0.983,
    1.046,
    0.964,
    1.004,
]


def normalize_latents(latent: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    """
    Normalize VAE latents with channel-wise standard deviations.

    Mochi VAE expects normalized latents as input. Each of the 12 latent
    channels has a specific standard deviation value.

    Args:
        latent: Input latent tensor of shape [B, 12, t, h, w]
        device: Target device for normalization tensor (defaults to latent.device)
        dtype: Target dtype for normalization tensor (defaults to latent.dtype)

    Returns:
        Normalized latent tensor of same shape as input
    """
    if device is None:
        device = latent.device
    if dtype is None:
        dtype = latent.dtype

    vae_std = torch.tensor(VAE_STD_CHANNELS, dtype=dtype, device=device)
    # Reshape to [1, 12, 1, 1, 1] for broadcasting
    vae_std = vae_std.view(1, 12, 1, 1, 1)

    return latent / vae_std
