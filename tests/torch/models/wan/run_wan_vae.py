#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone e2e smoke test: Wan VAE (encoder + decoder) on TT device.

Loads AutoencoderKLWan, compiles encoder and decoder separately with
torch.compile(backend="tt"), runs forward passes, and prints output statistics.

Usage:
    python tests/torch/models/wan/run_wan_vae.py
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

LATENT_CHANNELS = 16
LATENT_DEPTH = 2
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


def run_component(name, model, inputs, device):
    """Compile and run a single VAE component (encoder or decoder)."""
    print(f"\n--- {name} ---")

    # Move to device
    print(f"[device] Moving {name} to TT device ...")
    t0 = time.time()
    model = model.to(device)
    print(f"[device] Done in {time.time() - t0:.1f}s")

    # Compile
    print(f"[compile] torch.compile(backend='tt') ...")
    t0 = time.time()
    compiled = torch.compile(model, backend="tt")
    print(f"[compile] torch.compile returned in {time.time() - t0:.1f}s")

    # Forward pass
    print(f"[run] Running forward pass ...")
    inputs_dev = inputs.to(device)

    t0 = time.time()
    with torch.no_grad():
        output = compiled(inputs_dev)
    torch_xla.sync(wait=True)
    elapsed = time.time() - t0
    print(f"[run] Forward pass completed in {elapsed:.2f}s")

    # Results
    if isinstance(output, tuple):
        output = output[0]
    out_cpu = output.cpu()
    print(f"[result] output shape: {out_cpu.shape}")
    print(f"[result] dtype: {out_cpu.dtype}")
    print(f"[result] mean: {out_cpu.float().mean().item():.6f}")
    print(f"[result] std:  {out_cpu.float().std().item():.6f}")
    print(f"[result] min:  {out_cpu.float().min().item():.6f}")
    print(f"[result] max:  {out_cpu.float().max().item():.6f}")
    print(f"[result] has NaN: {out_cpu.isnan().any().item()}")
    print(f"[result] has Inf: {out_cpu.isinf().any().item()}")

    return out_cpu


def main():
    print("=" * 70)
    print("WAN VAE (encoder + decoder) — e2e smoke test")
    print("=" * 70)

    # ---- Device setup ----
    xr.set_device_type("TT")
    device = torch_xla.device()
    print(f"[setup] TT device: {device}")

    # ---- Load VAE ----
    print(f"\n[load] Loading VAE from {MODEL_ID} ...")
    t0 = time.time()
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )
    vae.eval()
    num_params = sum(p.numel() for p in vae.parameters())
    print(f"[load] Done in {time.time() - t0:.1f}s — {num_params / 1e6:.1f}M params")

    # ---- Prepare inputs ----
    # Decoder input: latent tensor
    decoder_input = torch.randn(
        1,
        LATENT_CHANNELS,
        LATENT_DEPTH,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        dtype=torch.float32,
    )
    # Encoder input: RGB video (T = 1 + 4*N for Wan temporal constraint)
    num_frames = 1 + 4 * LATENT_DEPTH  # 9
    encoder_input = torch.randn(
        1,
        3,
        num_frames,
        LATENT_HEIGHT * 8,
        LATENT_WIDTH * 8,
        dtype=torch.float32,
    )
    print(f"[input] decoder input: {decoder_input.shape}")
    print(f"[input] encoder input: {encoder_input.shape}")

    # ---- Run encoder ----
    run_component("VAE ENCODER", vae.encoder, encoder_input, device)

    # ---- Run decoder ----
    # Need fresh device state — reload decoder separately to avoid device conflicts
    # after encoder compilation.
    run_component("VAE DECODER", vae.decoder, decoder_input, device)

    print("\n" + "=" * 70)
    print("PASS — VAE encoder + decoder compiled and ran e2e on TT device")
    print("=" * 70)


if __name__ == "__main__":
    main()
