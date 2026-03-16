#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone e2e smoke test: Wan 1.3B transformer on TT device.

Loads the WanTransformer3DModel, compiles with torch.compile(backend="tt"),
runs a forward pass with small latent inputs, and prints output shape + statistics.

**Workaround**: The sinusoidal timestep embedding (sin/cos in
get_timestep_embedding) triggers an SFPI compiler ICE on Blackhole.
We pre-compute the timestep embedding on CPU and pass it into a wrapper
that bypasses the sin/cos operations in the compiled graph.

Usage:
    python tests/torch/models/wan/run_wan_transformer.py
"""

import math
import time

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
from diffusers.models.embeddings import get_timestep_embedding

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Small test dimensions — 32 patch tokens (2×4×4 after patchification)
LATENT_CHANNELS = 16
LATENT_DEPTH = 2
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
TEXT_DIM = 4096
TEXT_SEQ_LEN = 32


class WanTransformerNoSinCos(nn.Module):
    """Wrapper around WanTransformer3DModel that avoids sin/cos on device.

    Accepts a pre-computed timestep embedding (float) instead of a raw integer
    timestep, so the compiled graph never needs trigonometric SFPU kernels.

    The original forward path is replicated here minus the timesteps_proj call.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep_emb, encoder_hidden_states):
        """
        Args:
            hidden_states: [B, C, T, H, W] noisy latent video
            timestep_emb: [B, freq_dim] pre-computed sinusoidal embedding
            encoder_hidden_states: [B, seq, text_dim] text encoder output
        """
        t = self.transformer

        # 1. Shapes & RoPE (no sin/cos — buffers are pre-computed)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = t.config.patch_size
        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w

        rotary_emb = t.rope(hidden_states)

        # 2. Patch embedding
        hidden_states = t.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 3. Condition embedder — skip timesteps_proj (sin/cos)
        ce = t.condition_embedder
        time_embedder_dtype = next(iter(ce.time_embedder.parameters())).dtype
        if (
            timestep_emb.dtype != time_embedder_dtype
            and time_embedder_dtype != torch.int8
        ):
            timestep_emb = timestep_emb.to(time_embedder_dtype)
        temb = ce.time_embedder(timestep_emb).type_as(encoder_hidden_states)
        timestep_proj = ce.time_proj(ce.act_fn(temb))
        encoder_hidden_states = ce.text_embedder(encoder_hidden_states)

        # 4. Reshape timestep_proj → [B, 6, inner_dim]
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 5. Transformer blocks
        for block in t.blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )

        # 6. Output norm & projection
        shift, scale = (t.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(
            2, dim=1
        )
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (
            t.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = t.proj_out(hidden_states)

        # 7. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, ppf, pph, ppw, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output


def precompute_timestep_emb(timestep, freq_dim=256):
    """Compute sinusoidal timestep embedding on CPU (avoids SFPI sin/cos bug).

    Matches: Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    """
    return get_timestep_embedding(
        timestep,
        freq_dim,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
    )


def main():
    print("=" * 70)
    print("WAN TRANSFORMER (1.3B) — e2e smoke test (sin/cos workaround)")
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
    freq_dim = transformer.config.freq_dim
    print(f"[load] Done in {time.time() - t0:.1f}s — {num_params / 1e9:.2f}B params")
    print(f"[load] freq_dim={freq_dim}")

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

    # Pre-compute timestep embedding on CPU (workaround for SFPI sin/cos bug)
    timestep_emb = precompute_timestep_emb(timestep, freq_dim=freq_dim)
    print(
        f"[input] hidden_states: {hidden_states.shape}, "
        f"timestep_emb: {timestep_emb.shape} (pre-computed on CPU), "
        f"encoder_hidden_states: {encoder_hidden_states.shape}"
    )
    patch_tokens = (LATENT_DEPTH // 1) * (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)
    print(f"[input] → {patch_tokens} patch tokens after patchification")

    # ---- Wrap model ----
    print("\n[workaround] Wrapping transformer with WanTransformerNoSinCos ...")
    wrapper = WanTransformerNoSinCos(transformer)
    wrapper.eval()

    # ---- Move to device ----
    print("[device] Moving model to TT device ...")
    t0 = time.time()
    wrapper = wrapper.to(device)
    print(f"[device] Done in {time.time() - t0:.1f}s")

    # ---- Compile ----
    print("[compile] torch.compile(backend='tt') ...")
    t0 = time.time()
    compiled = torch.compile(wrapper, backend="tt")
    print(f"[compile] torch.compile returned in {time.time() - t0:.1f}s")

    # ---- Forward pass ----
    print("\n[run] Running forward pass ...")
    hs_dev = hidden_states.to(device)
    te_dev = timestep_emb.to(device)
    enc_dev = encoder_hidden_states.to(device)

    t0 = time.time()
    with torch.no_grad():
        output = compiled(hs_dev, te_dev, enc_dev)
    torch_xla.sync(wait=True)
    elapsed = time.time() - t0
    print(f"[run] Forward pass completed in {elapsed:.2f}s")

    # ---- Results ----
    sample_cpu = output.cpu()
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
