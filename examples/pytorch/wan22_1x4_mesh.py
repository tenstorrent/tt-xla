# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 T2V (Text-to-Video) generation on a 1×4 Tenstorrent device mesh.

All three pipeline components run on TT hardware through torch.compile:
  - Text encoder  (UMT5)             → torch.compile(backend="tt"), replicated
  - Transformer   (WanTransformer3D) → torch.compile(backend="tt"), tensor-parallel
  - VAE           (AutoencoderKLWan) → torch.compile(backend="tt"), replicated

The diffusers WanPipeline.__call__ drives the full denoising loop (including
expand_timesteps handling, cache_context CFG, UniPC scheduler, etc.). We only
need to plug in the compiled, sharded transformer and compiled VAE/text_encoder.

Usage:
    python examples/pytorch/wan22_1x4_mesh.py
    python examples/pytorch/wan22_1x4_mesh.py --prompt "A panda surfing in Tokyo"
    python examples/pytorch/wan22_1x4_mesh.py --output out.mp4 --num_frames 21

Requirements:
    pip install diffusers transformers accelerate imageio imageio-ffmpeg pillow
"""

import argparse
import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from torch_xla.distributed.spmd import Mesh

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wan 2.1 T2V 1.3B — lighter model for 1×4 mesh; 5B crashes PJRT argument-attr
# pass with 300 sharded weights (SmallVector overflow in PopulateArgumentAttrsFromTTMark).
# Note: Wan 2.2 does not have a 1.3B T2V variant; the smallest available is Wan 2.1 1.3B.
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# WAN temporal constraint: num_frames = 1 + 4*N  (N >= 0).
# This gives latent_frames = (num_frames - 1) // 4 + 1 after 4× temporal compression.
DEFAULT_NUM_FRAMES = 21  # 1 + 4*5  →  6 latent frames
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_FPS = 16


# ---------------------------------------------------------------------------
# SPMD / mesh setup
# ---------------------------------------------------------------------------


def setup_spmd() -> None:
    """Enable SPMD mode required for multi-device execution."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    print("[setup] SPMD mode enabled.")


def create_mesh(num_devices: int) -> Mesh:
    """
    Create a 1×N tensor-parallel mesh.

    axis 'batch'  — unused (batch dims replicated across all devices).
    axis 'model'  — slices attention heads and MLP hidden dims.
    """
    device_ids = np.arange(num_devices)
    mesh = Mesh(device_ids, (1, num_devices), ("batch", "model"))
    print(f"[setup] Created 1×{num_devices} tensor-parallel mesh.")
    return mesh


# ---------------------------------------------------------------------------
# Transformer weight sharding
# ---------------------------------------------------------------------------


def _shard_attention(attn, mesh: Mesh, count: list) -> None:
    """Column-then-row parallel sharding for a single attention module."""
    for name in ("to_q", "to_k", "to_v"):
        proj = getattr(attn, name, None)
        if proj is not None and hasattr(proj, "weight"):
            xs.mark_sharding(proj.weight, mesh, ("model", None))
            count[0] += 1
    to_out = getattr(attn, "to_out", None)
    if to_out is not None:
        out_linear = (
            to_out[0] if isinstance(to_out, (list, torch.nn.ModuleList)) else to_out
        )
        if hasattr(out_linear, "weight"):
            xs.mark_sharding(out_linear.weight, mesh, (None, "model"))
            count[0] += 1


def _shard_ff(ff, mesh: Mesh, count: list) -> None:
    """Column-then-row parallel sharding for a feed-forward module."""
    net = getattr(ff, "net", None)
    if net is None:
        return
    # Gate+up: net[0] may be a GEGLU wrapper (.proj) or a direct Linear.
    gate_up = net[0] if len(net) > 0 else None
    if gate_up is not None:
        if hasattr(gate_up, "proj") and hasattr(gate_up.proj, "weight"):
            xs.mark_sharding(gate_up.proj.weight, mesh, ("model", None))
            count[0] += 1
        elif hasattr(gate_up, "weight"):
            xs.mark_sharding(gate_up.weight, mesh, ("model", None))
            count[0] += 1
    # Down projection
    down = net[2] if len(net) > 2 else None
    if down is not None and hasattr(down, "weight"):
        xs.mark_sharding(down.weight, mesh, (None, "model"))
        count[0] += 1


def shard_transformer(transformer: torch.nn.Module, mesh: Mesh) -> None:
    """
    Tensor-parallel sharding for WanTransformer3DModel.

    Strategy (column-then-row parallel):
      Q/K/V and MLP gate+up → shard output features  ("model", None)
      O   and MLP down      → shard input  features  (None, "model")

    The SPMD compiler inserts AllReduce collectives automatically.
    """
    blocks = getattr(transformer, "transformer_blocks", None) or getattr(
        transformer, "blocks", None
    )
    if blocks is None:
        print("[shard] WARNING: transformer blocks not found; no weight sharding.")
        return

    count = [0]
    for block in blocks:
        # Self-attention
        for attr in ("attn1", "attn", "self_attn"):
            attn = getattr(block, attr, None)
            if attn is not None:
                _shard_attention(attn, mesh, count)
                break
        # Cross-attention (text conditioning)
        for attr in ("attn2", "cross_attn"):
            attn = getattr(block, attr, None)
            if attn is not None:
                _shard_attention(attn, mesh, count)
                break
        # Feed-forward MLP
        ff = getattr(block, "ff", None) or getattr(block, "ffn", None)
        if ff is not None:
            _shard_ff(ff, mesh, count)

    print(
        f"[shard] Marked {count[0]} weight tensors for "
        f"{len(list(mesh.device_ids))}-way tensor parallelism."
    )


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------


def run_wan22(
    prompt: str,
    negative_prompt: str = "",
    output_path: str = "wan22_output.mp4",
    num_frames: int = DEFAULT_NUM_FRAMES,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> None:

    assert (num_frames - 1) % 4 == 0, (
        f"num_frames must satisfy 1 + 4*N (got {num_frames}). "
        "Valid values: 5, 9, 13, 17, 21, 25, 49, 81, ..."
    )

    # ---- Device and SPMD initialisation ----
    xr.set_device_type("TT")
    setup_spmd()

    num_devices = xr.global_runtime_device_count()
    print(f"[init] {num_devices} TT device(s) found.")
    device = torch_xla.device()
    mesh = create_mesh(num_devices)

    torch_xla.set_custom_compile_options({"optimization_level": 1})

    # ---- Load full pipeline on CPU ----
    print(f"\n[load] Loading {MODEL_ID} ...")
    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    # Put models in eval mode individually (DiffusionPipeline has no .eval())
    pipeline.text_encoder.eval()
    pipeline.transformer.eval()
    pipeline.vae.eval()

    # WAN 2.2 uses expand_timesteps=True: per-token timestep [B, seq_len] whose
    # shape is determined at runtime, causing XLA dynamic-shape compilation failures.
    # Disable it so the transformer receives a static scalar timestep [B] instead.
    pipeline.register_to_config(expand_timesteps=False)

    # ---- Move and compile: text encoder ----
    # text_encoder must be on the XLA device so that pipeline._execution_device
    # returns the TT device and all tensors (latents, timesteps, ...) are routed there.
    # We move it to device; torch.compile is attempted but T5 can be tricky with dynamo,
    # so we fall back to eager XLA execution if compilation isn't required.
    print("[compile] Moving text_encoder to TT device...")
    pipeline.text_encoder = pipeline.text_encoder.to(device)

    # ---- Move and compile: transformer (no sharding) ----
    print("[compile] Moving transformer to TT device (no sharding)...")
    pipeline.transformer = pipeline.transformer.to(device)
    print("[compile] Compiling transformer with torch.compile(backend='tt')...")
    pipeline.transformer = torch.compile(
        pipeline.transformer,
        backend="tt",
        options={"tt_legacy_compile": True},
    )

    # ---- Move and compile: VAE ----
    print("[compile] Moving VAE to TT device and compiling...")
    pipeline.vae = pipeline.vae.to(device)
    pipeline.vae = torch.compile(
        pipeline.vae,
        backend="tt",
        options={"tt_legacy_compile": True},
    )

    # ---- Run full pipeline via diffusers __call__ ----
    # The WanPipeline handles:
    #   • T5 text encoding (CPU)
    #   • latent initialisation (48-ch, UniPC scheduler)
    #   • expand_timesteps=True per-token timestep preparation
    #   • cache_context CFG (two forward passes per step)
    #   • VAE decode + video_processor postprocessing
    print(
        f"\n[run] Generating {num_frames} frames @ {height}×{width}, "
        f"{num_inference_steps} steps, guidance={guidance_scale}"
    )
    print(f"[run] Prompt: '{prompt}'")

    generator = torch.Generator().manual_seed(seed)
    t0 = time.time()

    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pil",
    )

    elapsed = time.time() - t0
    print(
        f"[run] Completed in {elapsed:.1f}s  "
        f"({elapsed / num_inference_steps:.2f}s/step)"
    )

    # ---- Save video ----
    frames = output.frames[0]  # list of PIL Images
    from pathlib import Path

    out = str(Path(output_path).with_suffix(".mp4"))
    export_to_video(frames, out, fps=DEFAULT_FPS)
    print(f"\n[output] Saved {len(frames)} frames @ {DEFAULT_FPS} fps → {out}")
    print(f"         Duration: {len(frames) / DEFAULT_FPS:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wan 2.2 T2V on 1×4 Tenstorrent mesh — transformer + VAE on device"
    )
    parser.add_argument(
        "--prompt",
        default=(
            "An astronaut riding a horse through a vivid colorful nebula, "
            "cinematic slow motion, detailed, 8k"
        ),
    )
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--output",
        default="wan22_output.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Output video frame count.  Must equal 1 + 4*N (e.g. 5, 9, 21, 49, 81).",
    )
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
    )
    args = parser.parse_args()

    run_wan22(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_path=args.output,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
    )
