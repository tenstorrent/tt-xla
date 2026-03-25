# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro: Single MochiResnetBlock3D — unsharded vs sharded.

Runs the same ResBlock twice:
  1. Unsharded (baseline — should pass)
  2. Sharded with Megatron column-row tensor parallelism (hits layout bug)

The purpose is to produce two logs for comparison to identify exactly
where the Conv3d ROW_MAJOR → TILE layout mismatch occurs in the compiler.

Usage:
    # Unsharded (should work):
    python res_block.py --mode unsharded 2>&1 | tee res_block_unsharded.log

    # Sharded (expected to fail with Layout mismatch):
    python res_block.py --mode sharded 2>&1 | tee res_block_sharded.log
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import AutoencoderKLMochi
from diffusers.models.autoencoders.autoencoder_kl_mochi import MochiResnetBlock3D
from torch_xla.distributed.spmd import Mesh

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "0"
os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TT_RUNTIME_MEMORY_LOG_LEVEL"] = "operation"


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def _get_mesh():
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    if num_devices == 4:
        mesh_shape = (1, 4)
    elif num_devices == 8:
        mesh_shape = (4, 2)
    elif num_devices == 32:
        mesh_shape = (8, 4)
    else:
        raise ValueError(f"Unsupported device count: {num_devices}")
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"[Mesh] shape={mesh_shape} devices={num_devices}")
    return mesh


def _get_shard_axis(mesh):
    max_idx = max(range(len(mesh.mesh_shape)), key=lambda i: mesh.mesh_shape[i])
    return mesh.axis_names[max_idx]


# ---------------------------------------------------------------------------
# Sharding (Megatron column-row on a single ResBlock)
# ---------------------------------------------------------------------------
def _shard_resblock(block, mesh):
    """
    Apply Megatron column-row sharding to one MochiResnetBlock3D.

    conv1: column-parallel (shard C_out)
    norm2: sharded GroupNorm weight/bias to match channel-partitioned activations
    conv2: row-parallel (shard C_in)
    """
    axis = _get_shard_axis(mesh)

    # 5D Conv3d weight: [C_out, C_in, kT, kH, kW]
    COL_WEIGHT = (axis, None, None, None, None)
    ROW_WEIGHT = (None, axis, None, None, None)
    COL_BIAS = (axis,)
    NORM_WB = (axis,)

    # Conv1 — column-parallel
    xs.mark_sharding(block.conv1.conv.weight, mesh, COL_WEIGHT)
    xs.mark_sharding(block.conv1.conv.bias, mesh, COL_BIAS)

    # GroupNorm between conv1 and conv2
    xs.mark_sharding(block.norm2.norm_layer.weight, mesh, NORM_WB)
    xs.mark_sharding(block.norm2.norm_layer.bias, mesh, NORM_WB)

    # Conv2 — row-parallel
    xs.mark_sharding(block.conv2.conv.weight, mesh, ROW_WEIGHT)
    # conv2 bias NOT sharded (applied after implicit all-reduce)

    print(f"[Shard] ResBlock sharded on axis '{axis}'")


# ---------------------------------------------------------------------------
# Extract a single ResBlock from the pretrained VAE
# ---------------------------------------------------------------------------
def _load_single_resblock():
    """Load the first ResBlock from block_in (768 channels, 3x3x3 CausalConv3d)."""
    print("Loading AutoencoderKLMochi to extract single ResBlock...")
    t0 = time.time()
    vae = AutoencoderKLMochi.from_pretrained(
        "genmo/mochi-1-preview",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    # block_in is MochiMidBlock3D with 3 ResBlocks at 768 channels
    resblock = vae.decoder.block_in.resnets[0]
    print(f"Loaded ResBlock (768ch) in {time.time() - t0:.1f}s")
    print(f"  conv1: {resblock.conv1.conv.weight.shape}")
    print(f"  conv2: {resblock.conv2.conv.weight.shape}")
    print(f"  norm1 groups: {resblock.norm1.norm_layer.num_groups}")
    print(f"  norm2 groups: {resblock.norm2.norm_layer.num_groups}")
    return resblock


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------
def run_unsharded(resblock):
    """Run a single ResBlock WITHOUT sharding. Baseline that should work."""
    print("\n" + "=" * 60)
    print("MODE: UNSHARDED (baseline)")
    print("=" * 60)

    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    device = torch_xla.device()

    resblock = resblock.eval().to(device)
    compiled = torch.compile(resblock, backend="tt")

    # Input shape: [1, 768, 8, 60, 106] — matches block_in's first ResBlock
    # Using smaller spatial dims to reduce memory and speed up the test
    x = torch.randn(1, 768, 2, 8, 8, dtype=torch.bfloat16, device=device)

    print(f"Input shape: {x.shape}")
    print("Running forward pass (unsharded)...")
    t0 = time.time()
    with torch.no_grad():
        output = compiled(x)
    torch_xla.sync()
    print(f"Forward pass completed in {time.time() - t0:.1f}s")

    if isinstance(output, tuple):
        out_tensor = output[0]
    else:
        out_tensor = output
    print(f"Output shape: {out_tensor.shape}")
    assert (
        out_tensor.shape == x.shape
    ), f"Shape mismatch: {out_tensor.shape} vs {x.shape}"
    print("UNSHARDED: PASSED")


def run_sharded(resblock):
    """Run a single ResBlock WITH Megatron column-row sharding. Expected to fail."""
    print("\n" + "=" * 60)
    print("MODE: SHARDED (Megatron column-row)")
    print("=" * 60)

    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    device = torch_xla.device()
    mesh = _get_mesh()

    resblock = resblock.eval().to(device)
    compiled = torch.compile(resblock, backend="tt")

    # Apply sharding
    _shard_resblock(compiled, mesh)

    # Same input shape
    x = torch.randn(1, 768, 2, 8, 8, dtype=torch.bfloat16, device=device)

    print(f"Input shape: {x.shape}")
    print("Running forward pass (sharded)...")
    t0 = time.time()
    with torch.no_grad():
        output = compiled(x)
    torch_xla.sync()
    print(f"Forward pass completed in {time.time() - t0:.1f}s")

    if isinstance(output, tuple):
        out_tensor = output[0]
    else:
        out_tensor = output
    print(f"Output shape: {out_tensor.shape}")
    assert (
        out_tensor.shape == x.shape
    ), f"Shape mismatch: {out_tensor.shape} vs {x.shape}"
    print("SHARDED: PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal repro: MochiResnetBlock3D unsharded vs sharded"
    )
    parser.add_argument(
        "--mode",
        choices=["unsharded", "sharded"],
        required=True,
        help="Run mode: 'unsharded' (baseline) or 'sharded' (triggers layout bug)",
    )
    args = parser.parse_args()

    resblock = _load_single_resblock()

    if args.mode == "unsharded":
        run_unsharded(resblock)
    else:
        run_sharded(resblock)
