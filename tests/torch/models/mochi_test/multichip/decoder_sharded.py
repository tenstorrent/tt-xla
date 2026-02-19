# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi VAE Decoder with Spatial Input Sharding.

Runs the full VAE decoder (~362M params) with input spatially sharded across
4 TT devices along the height dimension. Model weights are replicated on all
devices (Conv3D + GroupNorm layers don't decompose well for tensor parallelism).

Usage:
    python tests/torch/models/mochi_test/multichip/decoder_sharded.py

Sharding strategy:
    - Model weights: REPLICATED on all devices (no mark_sharding on weights)
    - Input: spatially sharded on HEIGHT dimension (dim 3)
    - Mesh: (1, 4) with 4 devices, axis "spatial"

Input: Normalized VAE latents [1, 12, 4, 60, 106]
    - 12 latent channels
    - 4 temporal frames (→ 24 video frames after 6x temporal expansion)
    - 60 latent height (→ 480 pixels after 8x spatial expansion)
    - 106 latent width (→ 848 pixels after 8x spatial expansion)

Output: Video frames [1, 3, 24, 480, 848]

Why spatial sharding (not tensor parallel):
    - Conv3D weights have kernel spatial dims [out_ch, in_ch, kT, kH, kW] — splitting
      output channels creates cross-device dependencies at kernel boundaries
    - GroupNorm requires ALL channels within a group to be on the same device
    - The VAE is "only" 362M params (724MB in bf16) — weights fit on each device
    - Spatial sharding distributes the large activation tensors across devices

Why 4 devices (not 8):
    - Height 60 is divisible by 4 (15 per device) but not by 8 (7.5)
    - Through upsampling: 60→120→240→480, all divisible by 4
"""

import os
import sys
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import AutoencoderKLMochi
from torch_xla.distributed.spmd import Mesh

# Import patch_padding from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from patch_padding import replace_padding_to_constant

# Channel-wise standard deviations for VAE latent normalization
# Source: Mochi VAE implementation (12 latent channels)
VAE_STD_CHANNELS = [
    0.925, 0.934, 0.946, 0.939, 0.961, 1.033,
    0.979, 1.024, 0.983, 1.046, 0.964, 1.004,
]


def normalize_latents(latent, device=None, dtype=None):
    """
    Normalize VAE latents with channel-wise standard deviations.

    Each of the 12 latent channels has a specific standard deviation.
    The VAE expects latents divided by these values.

    Args:
        latent: Input tensor [B, 12, t, h, w]
        device: Target device (defaults to latent.device)
        dtype: Target dtype (defaults to latent.dtype)

    Returns:
        Normalized latent tensor
    """
    if device is None:
        device = latent.device
    if dtype is None:
        dtype = latent.dtype

    vae_std = torch.tensor(VAE_STD_CHANNELS, dtype=dtype, device=device)
    vae_std = vae_std.view(1, 12, 1, 1, 1)
    return latent / vae_std


def calculate_output_shape(latent_shape):
    """
    Calculate expected decoder output shape from latent input shape.

    Decoder expands: temporal 6x, spatial 8x8.
    [B, 12, t, h, w] → [B, 3, t*6, h*8, w*8]
    """
    return (
        latent_shape[0],  # batch
        3,                # RGB channels
        latent_shape[2] * 6,  # temporal expansion
        latent_shape[3] * 8,  # height expansion
        latent_shape[4] * 8,  # width expansion
    )


def run_vae_decoder_sharded():
    """
    Run Mochi VAE decoder with spatial input sharding.

    Input: [1, 12, 4, 60, 106] (latents for ~1s video at 480x848)
    Output: [1, 3, 24, 480, 848] (decoded video frames)
    """
    # ---- Setup ----
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    print(f"Number of TT devices: {num_devices}")

    # Use up to 4 devices for spatial sharding (height 60 is divisible by 4)
    num_spatial_devices = min(num_devices, 4)
    print(f"Using {num_spatial_devices} devices for spatial sharding")

    if num_spatial_devices < 2:
        raise RuntimeError(
            f"Spatial sharding requires at least 2 devices, got {num_devices}. "
            "This script is designed for multi-chip execution."
        )

    device = torch_xla.device()

    # ---- Create Mesh ----
    device_ids = np.arange(num_spatial_devices)
    mesh = Mesh(device_ids, mesh_shape=(1, num_spatial_devices), axis_names=("batch", "spatial"))
    print(f"Created mesh: shape=(1, {num_spatial_devices}), axes=('batch', 'spatial')")

    # ---- Load Model ----
    print("Loading AutoencoderKLMochi (~362M params)...")
    t0 = time.time()
    vae = AutoencoderKLMochi.from_pretrained(
        "genmo/mochi-1-preview",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    print(f"VAE loaded in {time.time() - t0:.1f}s")

    # Extract just the decoder (not the full VAE with encode/decode methods)
    decoder = vae.decoder

    # ---- Patch Padding ----
    # Replace replicate padding with constant padding to avoid L1 memory errors
    # from complex gather/embedding lowering in the compiler
    print("\nPatching CogVideoXCausalConv3d padding modes...")
    patched = replace_padding_to_constant(decoder)
    print(f"Patched {patched} layers\n")

    # Move to XLA device
    # Weights are NOT sharded — they stay replicated on all devices
    decoder = decoder.eval().to(device)

    # ---- Compile ----
    print("Compiling with TT backend...")
    t0 = time.time()
    compiled_decoder = torch.compile(decoder, backend="tt")
    print(f"torch.compile() returned in {time.time() - t0:.1f}s")

    # ---- Prepare Input ----
    # Realistic latent for ~1s video at 480x848 (24fps):
    #   [B, C, T, H, W] = [1, 12, 4, 60, 106]
    #   T = 24 frames / 6 temporal compression = 4
    #   H = 480 pixels / 8 spatial compression = 60
    #   W = 848 pixels / 8 spatial compression = 106
    latent_shape = (1, 12, 4, 60, 106)
    latent = torch.randn(*latent_shape, dtype=torch.bfloat16)

    # Normalize with channel-wise standard deviations (required by Mochi VAE)
    latent_normalized = normalize_latents(latent)
    latent_normalized = latent_normalized.to(device)

    # Mark spatial sharding on height dimension (dim 3)
    # Height 60 / 4 devices = 15 per device
    xs.mark_sharding(latent_normalized, mesh, (None, None, None, "spatial", None))
    print(f"Input shape: {latent_shape}, sharded on height: 60 / {num_spatial_devices} = {60 // num_spatial_devices} per device")

    # ---- Run Forward Pass ----
    print("Running forward pass...")
    t0 = time.time()
    with torch.no_grad():
        # Decoder returns (output_tensor, something_else) as a tuple
        output = compiled_decoder(latent_normalized)

    torch_xla.sync()
    elapsed = time.time() - t0
    print(f"Forward pass completed in {elapsed:.1f}s")

    # ---- Validate Output ----
    # Decoder output is a tuple: (decoded_video, ...)
    if isinstance(output, tuple):
        output_tensor = output[0]
    else:
        output_tensor = output

    expected_shape = calculate_output_shape(latent_shape)
    print(f"Output shape: {output_tensor.shape}, expected: {expected_shape}")

    assert output_tensor.shape == expected_shape, (
        f"Output shape mismatch: got {output_tensor.shape}, expected {expected_shape}"
    )

    print("VAE decoder spatial sharding test PASSED")


if __name__ == "__main__":
    print("=" * 70)
    print("Mochi VAE Decoder — Spatial Input Sharding")
    print("=" * 70)
    run_vae_decoder_sharded()
