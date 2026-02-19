# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi DiT (Diffusion Transformer) with Megatron-style Tensor Parallelism.

Runs the full 48-layer MochiTransformer3DModel (~10B params) sharded across
8 TT devices using tensor parallelism. Requires a QuietBox (4x p150b cards).

Usage:
    python tests/torch/models/mochi_test/multichip/dit_sharded.py

Sharding strategy (Megatron-style):
    - Attention Q/K/V projections: column-parallel ("model", None)
    - Attention output projection: row-parallel (None, "model")
    - MLP gate+up (SwiGLU): column-parallel ("model", None)
    - MLP down projection: row-parallel (None, "model")
    - Inputs: replicated across all devices

Mochi DiT specs:
    - 48 MochiTransformerBlocks (AsymmDiT)
    - 24 attention heads, 128 dim/head → inner_dim = 3072
    - Visual stream: 3072-dim, text stream: 1536-dim (upscaled to 3072 for attention)
    - SwiGLU MLP with 8x expansion (visual) and ~5.3x expansion (text)

With 8 devices: 24 heads / 8 = 3 heads per device, all dims divisible by 8.
"""

import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import MochiTransformer3DModel
from torch_xla.distributed.spmd import Mesh


def apply_dit_tensor_parallel(model, mesh):
    """
    Apply Megatron-style tensor parallel sharding to Mochi DiT.

    For each of the 48 transformer blocks, shards:
    - Visual + text attention: QKV column-parallel, output row-parallel
    - Visual + text MLP: gate+up column-parallel, down row-parallel

    The last block (context_pre_only=True) has no text MLP (ff_context=None),
    but still has text attention projections.

    Args:
        model: MochiTransformer3DModel instance (already on XLA device)
        mesh: Mesh object with "model" axis for tensor parallelism
    """
    shard_specs = {}

    for i, block in enumerate(model.transformer_blocks):
        # ---- Visual Attention ----
        # Q/K/V: column-parallel (split output dim across devices)
        shard_specs[block.attn1.to_q.weight] = ("model", None)
        shard_specs[block.attn1.to_k.weight] = ("model", None)
        shard_specs[block.attn1.to_v.weight] = ("model", None)
        # Output: row-parallel (split input dim, ALL-REDUCE after)
        # bias=True on to_out[0], but we leave bias replicated
        shard_specs[block.attn1.to_out[0].weight] = (None, "model")

        # ---- Text Attention ----
        # Q/K/V: column-parallel (1536→3072, split the 3072 output dim)
        shard_specs[block.attn1.add_q_proj.weight] = ("model", None)
        shard_specs[block.attn1.add_k_proj.weight] = ("model", None)
        shard_specs[block.attn1.add_v_proj.weight] = ("model", None)
        # Output: row-parallel (3072→1536, split the 3072 input dim)
        # to_add_out exists even on the last block
        if block.attn1.to_add_out is not None:
            shard_specs[block.attn1.to_add_out.weight] = (None, "model")

        # ---- Visual MLP (SwiGLU) ----
        # gate+up: column-parallel (Linear(3072, 16384) for SwiGLU)
        shard_specs[block.ff.net[0].proj.weight] = ("model", None)
        # down: row-parallel (Linear(8192, 3072))
        shard_specs[block.ff.net[2].weight] = (None, "model")

        # ---- Text MLP (SwiGLU) ----
        # Last block (context_pre_only=True) has ff_context=None
        if block.ff_context is not None:
            shard_specs[block.ff_context.net[0].proj.weight] = ("model", None)
            shard_specs[block.ff_context.net[2].weight] = (None, "model")

    # Apply all sharding specs
    sharded_count = 0
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
        sharded_count += 1

    print(f"Applied sharding to {sharded_count} weight tensors across {len(model.transformer_blocks)} blocks")


def run_dit_sharded():
    """
    Run Mochi DiT with tensor parallelism across all available TT devices.

    Input (realistic for ~1s video at 480x848, 24fps):
        - hidden_states: [1, 12, 4, 60, 106]  (latent video)
        - encoder_hidden_states: [1, 256, 4096] (T5-XXL text embeddings)
        - timestep: [500] (diffusion step)
        - encoder_attention_mask: [1, 256] (text mask)

    Output: [1, 12, 4, 60, 106] (denoised latent, same shape as input)
    """
    # ---- Setup ----
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    print(f"Number of TT devices: {num_devices}")

    if num_devices < 2:
        raise RuntimeError(
            f"Tensor parallelism requires at least 2 devices, got {num_devices}. "
            "This script is designed for multi-chip execution (8 devices on QuietBox)."
        )

    # Validate head divisibility
    num_heads = 24
    if num_heads % num_devices != 0:
        raise ValueError(
            f"Number of attention heads ({num_heads}) must be divisible by "
            f"number of devices ({num_devices}) for head-parallel sharding."
        )

    device = torch_xla.device()

    # ---- Create Mesh ----
    device_ids = np.arange(num_devices)
    mesh = Mesh(device_ids, mesh_shape=(1, num_devices), axis_names=("batch", "model"))
    print(f"Created mesh: shape=(1, {num_devices}), axes=('batch', 'model')")

    # ---- Load Model ----
    print("Loading MochiTransformer3DModel (~10B params, ~20GB in bf16)...")
    t0 = time.time()
    transformer = MochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Move to XLA device
    transformer = transformer.eval().to(device)

    # ---- Apply Tensor Parallel Sharding ----
    apply_dit_tensor_parallel(transformer, mesh)

    # ---- Compile ----
    print("Compiling with TT backend...")
    t0 = time.time()
    compiled_transformer = torch.compile(transformer, backend="tt")
    print(f"torch.compile() returned in {time.time() - t0:.1f}s")

    # ---- Prepare Inputs ----
    # Realistic input for ~1s video at 480x848 resolution (24fps):
    # Latent: [B, C, T, H, W] = [1, 12, 4, 60, 106]
    #   T = 24 frames / 6 temporal compression = 4
    #   H = 480 pixels / 8 spatial compression = 60
    #   W = 848 pixels / 8 spatial compression = 106 (rounded from 106)
    hidden_states = torch.randn(1, 12, 4, 60, 106, dtype=torch.bfloat16).to(device)

    # T5-XXL text embeddings: [B, seq_len, 4096]
    encoder_hidden_states = torch.randn(1, 256, 4096, dtype=torch.bfloat16).to(device)

    # Diffusion timestep (0-999 range)
    timestep = torch.tensor([500], dtype=torch.long).to(device)

    # Text attention mask (all tokens valid)
    encoder_attention_mask = torch.ones(1, 256, dtype=torch.long).to(device)

    # Mark inputs as replicated (no sharding on inputs for TP)
    xs.mark_sharding(hidden_states, mesh, (None, None, None, None, None))
    xs.mark_sharding(encoder_hidden_states, mesh, (None, None, None))
    xs.mark_sharding(encoder_attention_mask, mesh, (None, None))

    # ---- Run Forward Pass ----
    print("Running forward pass...")
    t0 = time.time()
    with torch.no_grad():
        output = compiled_transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
        )

    torch_xla.sync()
    elapsed = time.time() - t0
    print(f"Forward pass completed in {elapsed:.1f}s")

    # ---- Validate Output ----
    expected_shape = (1, 12, 4, 60, 106)
    output_tensor = output.sample
    print(f"Output shape: {output_tensor.shape}, expected: {expected_shape}")

    assert output_tensor.shape == expected_shape, (
        f"Output shape mismatch: got {output_tensor.shape}, expected {expected_shape}"
    )

    print("DiT tensor parallel test PASSED")


if __name__ == "__main__":
    print("=" * 70)
    print("Mochi DiT — Tensor Parallel (Megatron-style)")
    print("=" * 70)
    run_dit_sharded()
