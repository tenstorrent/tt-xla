#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone e2e smoke test: Wan UMT5-XXL text encoder on TT device.

Loads the text encoder, compiles with torch.compile(backend="tt"),
runs a forward pass, and prints output shape + basic statistics.

Usage:
    python tests/torch/models/wan/run_wan_text_encoder.py
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SEQ_LEN = 32
VOCAB_SIZE = 256384


def main():
    print("=" * 70)
    print("WAN TEXT ENCODER (UMT5-XXL) — e2e smoke test")
    print("=" * 70)

    # ---- Device setup ----
    xr.set_device_type("TT")
    device = torch_xla.device()
    print(f"[setup] TT device: {device}")

    # ---- Load model ----
    print(f"\n[load] Loading text encoder from {MODEL_ID} ...")
    t0 = time.time()
    from transformers import UMT5EncoderModel

    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float32
    )
    text_encoder.eval()
    num_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"[load] Done in {time.time() - t0:.1f}s — {num_params / 1e9:.2f}B params")

    # ---- Prepare inputs ----
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
    attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)
    print(
        f"[input] input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}"
    )

    # ---- Move to device ----
    print("\n[device] Moving model to TT device ...")
    t0 = time.time()
    text_encoder = text_encoder.to(device)
    print(f"[device] Done in {time.time() - t0:.1f}s")

    # ---- Compile ----
    print("[compile] torch.compile(backend='tt') ...")
    t0 = time.time()
    compiled = torch.compile(text_encoder, backend="tt")
    print(f"[compile] torch.compile returned in {time.time() - t0:.1f}s")

    # ---- Forward pass ----
    print("\n[run] Running forward pass ...")
    input_ids_dev = input_ids.to(device)
    attention_mask_dev = attention_mask.to(device)

    t0 = time.time()
    with torch.no_grad():
        output = compiled(input_ids=input_ids_dev, attention_mask=attention_mask_dev)
    torch_xla.sync(wait=True)
    elapsed = time.time() - t0
    print(f"[run] Forward pass completed in {elapsed:.2f}s")

    # ---- Results ----
    hidden = output.last_hidden_state
    hidden_cpu = hidden.cpu()
    print(f"\n[result] last_hidden_state shape: {hidden_cpu.shape}")
    print(f"[result] dtype: {hidden_cpu.dtype}")
    print(f"[result] mean: {hidden_cpu.float().mean().item():.6f}")
    print(f"[result] std:  {hidden_cpu.float().std().item():.6f}")
    print(f"[result] min:  {hidden_cpu.float().min().item():.6f}")
    print(f"[result] max:  {hidden_cpu.float().max().item():.6f}")
    print(f"[result] has NaN: {hidden_cpu.isnan().any().item()}")
    print(f"[result] has Inf: {hidden_cpu.isinf().any().item()}")

    print("\n" + "=" * 70)
    print("PASS — text encoder compiled and ran e2e on TT device")
    print("=" * 70)


if __name__ == "__main__":
    main()
