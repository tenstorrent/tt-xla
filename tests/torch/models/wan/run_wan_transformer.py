#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone e2e smoke test: Wan 1.3B transformer on TT device.

Loads the WanTransformer3DModel, compiles with torch.compile(backend="tt"),
runs a forward pass with small latent inputs, and prints output shape + statistics.

Usage:
    python tests/torch/models/wan/run_wan_transformer.py
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Small test dimensions — 32 patch tokens (2×4×4 after patchification)
LATENT_CHANNELS = 16
LATENT_DEPTH = 2
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
TEXT_DIM = 4096
TEXT_SEQ_LEN = 32


def main():
    print("=" * 70)
    print("WAN TRANSFORMER (1.3B) — e2e smoke test")
    print("=" * 70)

    # ---- Device setup ----
    xr.set_device_type("TT")
    device = torch_xla.device()
    print(f"[setup] TT device: {device}")

    # ---- Load model ----
    print(f"\n[load] Loading transformer from {MODEL_ID} ...")
    t0 = time.time()
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.float32
    )
    transformer.eval()
    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"[load] Done in {time.time() - t0:.1f}s — {num_params / 1e9:.2f}B params")

    # ---- Prepare inputs ----
    hidden_states = torch.randn(
        1,
        LATENT_CHANNELS,
        LATENT_DEPTH,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        dtype=torch.float32,
    )
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, TEXT_SEQ_LEN, TEXT_DIM, dtype=torch.float32)
    print(
        f"[input] hidden_states: {hidden_states.shape}, "
        f"timestep: {timestep.shape}, "
        f"encoder_hidden_states: {encoder_hidden_states.shape}"
    )
    patch_tokens = (LATENT_DEPTH // 1) * (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)
    print(f"[input] → {patch_tokens} patch tokens after patchification")

    # ---- Move to device ----
    print("\n[device] Moving model to TT device ...")
    t0 = time.time()
    transformer = transformer.to(device)
    print(f"[device] Done in {time.time() - t0:.1f}s")

    # ---- Compile ----
    print("[compile] torch.compile(backend='tt') ...")
    t0 = time.time()
    compiled = torch.compile(transformer, backend="tt")
    print(f"[compile] torch.compile returned in {time.time() - t0:.1f}s")

    # ---- Forward pass ----
    print("\n[run] Running forward pass ...")
    hs_dev = hidden_states.to(device)
    ts_dev = timestep.to(device)
    enc_dev = encoder_hidden_states.to(device)

    t0 = time.time()
    with torch.no_grad():
        output = compiled(
            hidden_states=hs_dev,
            timestep=ts_dev,
            encoder_hidden_states=enc_dev,
            return_dict=False,
        )
    torch_xla.sync(wait=True)
    elapsed = time.time() - t0
    print(f"[run] Forward pass completed in {elapsed:.2f}s")

    # ---- Results ----
    sample = output[0]
    sample_cpu = sample.cpu()
    print(f"\n[result] output sample shape: {sample_cpu.shape}")
    print(f"[result] dtype: {sample_cpu.dtype}")
    print(f"[result] mean: {sample_cpu.float().mean().item():.6f}")
    print(f"[result] std:  {sample_cpu.float().std().item():.6f}")
    print(f"[result] min:  {sample_cpu.float().min().item():.6f}")
    print(f"[result] max:  {sample_cpu.float().max().item():.6f}")
    print(f"[result] has NaN: {sample_cpu.isnan().any().item()}")
    print(f"[result] has Inf: {sample_cpu.isinf().any().item()}")

    print("\n" + "=" * 70)
    print("PASS — transformer compiled and ran e2e on TT device")
    print("=" * 70)


if __name__ == "__main__":
    main()
