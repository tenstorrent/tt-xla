# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi VAE Decoder — Megatron-style channel tensor-parallel sharding.

Sharding strategy:
    Each MochiResnetBlock3D contains two CausalConv3d layers and a GroupNorm
    between them. We apply Megatron-style column-row pairing:
        conv1: column-parallel (shard C_out across devices)
        norm2: channel-sharded weight/bias (8 groups/device, fully local)
        conv2: row-parallel (shard C_in across devices)
    The all-reduce after conv2 is inserted automatically by SPMD/Shardy.

    Boundary layers (conv_in, proj_out, unpatchify proj) are left unsharded.
    GroupNorm with 32 groups divides evenly across 4 devices (8 groups/device),
    so normalization requires zero cross-device communication.

    The pixel shuffle (unpatchify) in each MochiUpBlock3D is rewritten as a
    staged decomposition that merges one dimension pair at a time, keeping
    large dimensions in the tile-padded positions to avoid catastrophic tile
    padding overhead. See pixel_shuffle_problem.md for the full analysis.

    Total CCL ops: 19 all-reduces (1 per ResBlock, implicit via SPMD).

    See decoder_strategy.md for full analysis.

Usage:
    python tests/torch/models/mochi_test/multichip/decoder_sharded.py

Input: Normalized VAE latents [1, 12, 8, 60, 106]
    - 12 latent channels
    - 8 temporal frames (→ 48 video frames after 6x temporal expansion)
    - 60 latent height (→ 480 pixels after 8x spatial expansion)
    - 106 latent width (→ 848 pixels after 8x spatial expansion)

Output: Video frames [1, 3, 48, 480, 848]
"""

import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import AutoencoderKLMochi
from diffusers.models.autoencoders.autoencoder_kl_mochi import (
    MochiResnetBlock3D,
    MochiUpBlock3D,
)
from torch_xla.distributed.spmd import Mesh

# Set necessary environment variables
os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "0"
os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TT_RUNTIME_MEMORY_LOG_LEVEL"] = "operation"

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


# ---------------------------------------------------------------------------
# Mesh setup
# ---------------------------------------------------------------------------
def _get_mesh():
    """
    Create SPMD mesh for tensor-parallel sharding.
    Returns mesh with axes ("batch", "model"):
        QB (4 chips):      (1, 4) — 4-way model-parallel
        LLMBox (8 chips):  (4, 2)
        Galaxy (32 chips):  (8, 4)
    """
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    if num_devices == 32:
        mesh_shape = (8, 4)
    elif num_devices == 8:
        mesh_shape = (1, 8)
    elif num_devices == 4:
        mesh_shape = (1, 4)
    else:
        raise ValueError(
            f"Unsupported device count: {num_devices}. Expected 4, 8, or 32."
        )

    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"[Mesh] shape={mesh_shape} axes=('batch','model') devices={num_devices}")
    return mesh


def _get_shard_axis(mesh):
    """Return the name of the mesh axis with the most devices."""
    max_idx = max(range(len(mesh.mesh_shape)), key=lambda i: mesh.mesh_shape[i])
    return mesh.axis_names[max_idx]


# ---------------------------------------------------------------------------
# Pixel shuffle patch — staged decomposition to avoid tile padding blowup
# ---------------------------------------------------------------------------
def _patch_pixel_shuffle(decoder):
    """
    Monkey-patch MochiUpBlock3D.forward() to use a staged pixel shuffle
    decomposition that avoids placing small factors (st/sh/sw=2) in the
    last two tile-padded dimensions.

    The original pixel shuffle does:
        view 8D -> permute(0,1,5,2,6,3,7,4) -> view 5D
    which puts sw=2 as the last dim, causing 16-19x tile padding overhead
    (596 MB -> 11,520 MB for up_block_0).

    The staged version merges one dimension pair at a time, keeping large
    dims in the tile-padded positions (max overhead: 1.33x).

    See pixel_shuffle_problem.md for full analysis.
    """

    def _make_patched_forward(original_forward):
        import functools

        @functools.wraps(original_forward)
        def patched_forward(self, hidden_states, conv_cache=None):
            new_conv_cache = {}
            conv_cache = conv_cache or {}

            for i, resnet in enumerate(self.resnets):
                conv_cache_key = f"resnet_{i}"
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states, new_conv_cache[conv_cache_key] = (
                        self._gradient_checkpointing_func(
                            resnet,
                            hidden_states,
                            conv_cache.get(conv_cache_key),
                        )
                    )
                else:
                    hidden_states, new_conv_cache[conv_cache_key] = resnet(
                        hidden_states, conv_cache=conv_cache.get(conv_cache_key)
                    )

            # Linear projection (unchanged)
            hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
            hidden_states = self.proj(hidden_states)
            hidden_states = hidden_states.permute(0, 4, 1, 2, 3)

            B, C_packed, T, H, W = hidden_states.shape
            st = self.temporal_expansion
            sh = self.spatial_expansion
            sw = self.spatial_expansion
            C = C_packed // (st * sh * sw)

            # --- Staged pixel shuffle ---
            # Merges one dimension pair at a time, keeping large dims in
            # the last two tile-padded positions (max overhead: 1.33x).
            # No .contiguous() calls — intermediates stay as lazy views so
            # PyTorch/XLA doesn't return them as separate graph outputs.
            # See pixel_shuffle_problem.md and test_pixel_shuffle_rewrite.py.

            # Unpack channel dim
            hidden_states = hidden_states.view(B, C, st, sh, sw, T, H, W)
            #                                  0  1   2   3   4   5  6  7

            # Step 1: Bring T adjacent to st, keep (H, W) in tile positions
            hidden_states = hidden_states.permute(0, 1, 5, 2, 3, 4, 6, 7)
            # -> [B, C, T, st, sh, sw, H, W]   last2=(H,W) both large

            # Step 2: Merge (T, st) -> T*st  (t * st + dt = correct interleaving)
            hidden_states = hidden_states.reshape(B, C, T * st, sh, sw, H, W)
            # -> [B, C, T*st, sh, sw, H, W]    last2=(H,W) still large

            # Step 3: Move (H,sh) and (W,sw) adjacent, put (C,T*st) in tile positions
            hidden_states = hidden_states.permute(0, 5, 3, 6, 4, 1, 2)
            # -> [B, H, sh, W, sw, C, T*st]    last2=(C,T*st) both >= 24

            # Step 4: Merge (H,sh)->H*sh and (W,sw)->W*sw
            hidden_states = hidden_states.reshape(B, H * sh, W * sw, C, T * st)
            # -> [B, H*sh, W*sw, C, T*st]      last2=(C,T*st) still large

            # Step 5: Restore standard BCTHW layout
            hidden_states = hidden_states.permute(0, 3, 4, 1, 2)
            # -> [B, C, T*st, H*sh, W*sw]      last2=(H*sh,W*sw) both >= 120

            return hidden_states, new_conv_cache

        return patched_forward

    count = 0
    for up_block in decoder.up_blocks:
        up_block.forward = _make_patched_forward(up_block.forward).__get__(
            up_block, MochiUpBlock3D
        )
        count += 1

    print(
        f"[Patch] {count} MochiUpBlock3D pixel shuffles patched (staged decomposition)"
    )
    return count


# ---------------------------------------------------------------------------
# Divisibility pre-check
# ---------------------------------------------------------------------------
def _verify_divisibility(decoder, mesh):
    """
    Verify all MochiResnetBlock3D conv channels are divisible by shard axis.

    Column-parallel conv1 shards C_out; row-parallel conv2 shards C_in.
    Both must divide evenly by the number of devices on the shard axis.
    GroupNorm with 32 groups must also divide evenly (32 / axis_size).
    """
    axis = _get_shard_axis(mesh)
    axis_size = mesh.mesh_shape[mesh.axis_names.index(axis)]

    count = 0
    for name, m in decoder.named_modules():
        if not isinstance(m, MochiResnetBlock3D):
            continue

        # conv1 (column-parallel): shard C_out
        c_out = m.conv1.conv.weight.shape[0]
        assert (
            c_out % axis_size == 0
        ), f"{name}.conv1: C_out={c_out} not divisible by {axis}={axis_size}"

        # conv2 (row-parallel): shard C_in
        c_in = m.conv2.conv.weight.shape[1]
        assert (
            c_in % axis_size == 0
        ), f"{name}.conv2: C_in={c_in} not divisible by {axis}={axis_size}"

        # GroupNorm groups must also divide evenly
        num_groups = m.norm2.norm_layer.num_groups
        assert (
            num_groups % axis_size == 0
        ), f"{name}.norm2: num_groups={num_groups} not divisible by {axis}={axis_size}"

        count += 1

    print(
        f"[Verify] {count} MochiResnetBlock3D pass divisibility (axis '{axis}'={axis_size})"
    )
    return count


# ---------------------------------------------------------------------------
# Decoder weight sharding — Megatron column-row tensor parallelism
# ---------------------------------------------------------------------------
def _shard_decoder_weights(decoder, mesh):
    """
    Apply Megatron-style column→row tensor-parallel sharding to all
    MochiResnetBlock3D weights in the decoder.

    Per ResBlock (19 total):
        conv1 (CogVideoXCausalConv3d): column-parallel — shard C_out
        norm2 (MochiChunkedGroupNorm3D): channel-sharded — shard weight/bias
            GroupNorm with 32 groups / 4 devices = 8 groups/device.
            Each group normalizes independently, so no cross-device
            communication is needed. Weight and bias are per-channel
            parameters that must be sharded to match the channel-sharded
            activations.
        conv2 (CogVideoXCausalConv3d): row-parallel — shard C_in

    Attribute paths (verified from diffusers source):
        conv weight:  block.conv{1,2}.conv.weight  [C_out, C_in, kT, kH, kW]
        conv bias:    block.conv{1,2}.conv.bias     [C_out] or [C_in]
        norm weight:  block.norm2.norm_layer.weight  [C]
        norm bias:    block.norm2.norm_layer.bias     [C]

    Boundary layers left unsharded:
        - conv_in: nn.Conv3d(12→768, 1×1×1) — C_in=12 too small
        - proj_out: nn.Linear(128→3) — C_out=3 not divisible
        - up_blocks[j].proj: unpatchify linear — runs on replicated data
    """
    axis = _get_shard_axis(mesh)

    # Partition specs for 5D Conv3d weight [C_out, C_in, kD, kH, kW]
    COL_WEIGHT = (axis, None, None, None, None)  # shard C_out (column-parallel)
    ROW_WEIGHT = (None, axis, None, None, None)  # shard C_in (row-parallel)

    # Partition spec for 1D tensors [C] (bias, norm weight, norm bias)
    SHARD_C = (axis,)

    def _shard_resblock(block):
        """Apply Megatron column-row sharding to one MochiResnetBlock3D."""
        # Conv1 — column-parallel (shard output channels)
        xs.mark_sharding(block.conv1.conv.weight, mesh, COL_WEIGHT)
        xs.mark_sharding(block.conv1.conv.bias, mesh, SHARD_C)

        # GroupNorm between conv1 and conv2 — channel-sharded.
        # 32 groups / 4 devices = 8 groups/device, each with the same
        # channels-per-group as unsharded (e.g., 768/32=24). Normalization
        # is fully local — zero CCL ops.
        xs.mark_sharding(block.norm2.norm_layer.weight, mesh, SHARD_C)
        xs.mark_sharding(block.norm2.norm_layer.bias, mesh, SHARD_C)

        # Conv2 — row-parallel (shard input channels)
        xs.mark_sharding(block.conv2.conv.weight, mesh, ROW_WEIGHT)
        # conv2 bias is NOT sharded — applied after implicit all-reduce

    count = 0

    # block_in: MochiMidBlock3D with 3 ResBlocks
    for resblock in decoder.block_in.resnets:
        _shard_resblock(resblock)
        count += 1

    # up_blocks: 3 MochiUpBlock3D, each with N ResBlocks + unpatchify proj
    for up_block in decoder.up_blocks:
        for resblock in up_block.resnets:
            _shard_resblock(resblock)
            count += 1
        # up_block.proj (unpatchify linear) is left unsharded

    # block_out: MochiMidBlock3D with 3 ResBlocks
    for resblock in decoder.block_out.resnets:
        _shard_resblock(resblock)
        count += 1

    annotations_per_block = (
        4  # conv1 weight + conv1 bias + norm2 weight + norm2 bias + conv2 weight = 5
    )
    # (but conv2 bias is not sharded, so 4 mark_sharding + 1 conv2 weight = 5 per block)
    print(
        f"[Shard] {count} MochiResnetBlock3D sharded "
        f"(5 mark_sharding per block = {count * 5} total) "
        f"on axis '{axis}'"
    )
    return count


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def normalize_latents(latent, device=None, dtype=None):
    """
    Normalize VAE latents with channel-wise standard deviations.
    Each of the 12 latent channels is divided by its std.

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
        3,  # RGB channels
        latent_shape[2] * 6,  # temporal expansion
        latent_shape[3] * 8,  # height expansion
        latent_shape[4] * 8,  # width expansion
    )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def run_vae_decoder_sharded():
    """
    Run Mochi VAE decoder with Megatron-style channel tensor-parallel sharding.

    Input: [1, 12, 8, 60, 106] (latents for ~2s video at 480x848)
    Output: [1, 3, 48, 480, 848] (decoded video frames)
    """
    # ---- Setup ----
    torch_xla.set_custom_compile_options(
        {"experimental-enable-dram-space-saving-optimization": "true"}
    )
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    device = torch_xla.device()
    mesh = _get_mesh()

    # ---- Load Model ----
    print("Loading AutoencoderKLMochi (~362M params)...")
    t0 = time.time()
    vae = AutoencoderKLMochi.from_pretrained(
        "genmo/mochi-1-preview",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    print(f"VAE loaded in {time.time() - t0:.1f}s")

    # Extract the decoder
    decoder = vae.decoder

    # ---- Patch Pixel Shuffle ----
    # Replace the original view+permute+view with a staged decomposition
    # that avoids placing small factors (sw=2) in tile-padded dimensions.
    # Reduces peak intermediate from 11-80 GB to 0.8-6.4 GB per pixel shuffle.
    # _patch_pixel_shuffle(decoder)

    # ---- Verify Divisibility ----
    _verify_divisibility(decoder, mesh)

    # ---- Move to Device & Shard ----
    decoder = decoder.eval().to(device)
    print("Compiling with TT backend...")
    t0 = time.time()
    compiled_decoder = torch.compile(decoder, backend="tt")
    print(f"torch.compile() returned in {time.time() - t0:.1f}s")
    _shard_decoder_weights(compiled_decoder, mesh)

    # ---- Prepare Input ----
    # Latent for ~2s video at 480x848 (24fps):
    #   [B, C, T, H, W] = [1, 12, 8, 60, 106]
    #   T = 48 frames / 6 temporal compression = 8
    #   H = 480 pixels / 8 spatial compression = 60
    #   W = 848 pixels / 8 spatial compression = 106
    latent_shape = (1, 12, 8, 60, 106)
    latent = torch.randn(*latent_shape, dtype=torch.bfloat16)

    # Normalize with channel-wise standard deviations (required by Mochi VAE)
    latent_normalized = normalize_latents(latent)
    latent_normalized = latent_normalized.to(device)

    # ---- Run Forward Pass ----
    print("Running forward pass...")
    t0 = time.time()
    with torch.no_grad():
        output = compiled_decoder(latent_normalized)

    torch_xla.sync()
    elapsed = time.time() - t0
    print(f"Forward pass completed in {elapsed:.1f}s")

    # ---- Validate Output ----
    if isinstance(output, tuple):
        output_tensor = output[0]
    else:
        output_tensor = output

    expected_shape = calculate_output_shape(latent_shape)
    print(f"Output shape: {output_tensor.shape}, expected: {expected_shape}")

    assert (
        output_tensor.shape == expected_shape
    ), f"Output shape mismatch: got {output_tensor.shape}, expected {expected_shape}"

    print("VAE decoder Megatron channel-TP sharding test PASSED")


if __name__ == "__main__":
    print("=" * 70)
    print("Mochi VAE Decoder — Megatron Channel Tensor-Parallel Sharding")
    print("=" * 70)
    run_vae_decoder_sharded()
