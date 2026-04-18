#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone VAE decoder test — skips text encoder + transformer for fast iteration.

    python vae_test.py
    python vae_test.py --seed 0 --output vae_test.png
"""

import argparse

import torch
import ttnn
from diffusers.image_processor import VaeImageProcessor

from vae.model_ttnn import VaeDecoderTTNN

IMG_LATENT_H    = 64
IMG_LATENT_W    = 64
LATENT_CHANNELS = 16


def open_mesh_device():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 4)),
        l1_small_size=1 << 15,
        trace_region_size=50_000_000,
    )
    device.enable_program_cache()
    return device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="vae_test.png")
    args = ap.parse_args()

    print("[1/3] Opening TTNN (1,4) mesh device ...")
    mesh_device = open_mesh_device()

    print("[2/3] Loading TTNN VAE decoder ...")
    vae_tt = VaeDecoderTTNN(mesh_device)
    vae_processor = VaeImageProcessor(vae_scale_factor=16)

    print("[3/3] Running VAE decode ...")
    torch.manual_seed(args.seed)
    latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)

    for run_idx in range(2):
        print(f"── run {run_idx + 1}/2 ─────────────────────────────")
        image_tensor = vae_tt(latents)
        out_path = args.output if run_idx == 1 else args.output.replace(".png", f".run{run_idx}.png")
        image = vae_processor.postprocess(image_tensor, output_type="pil")[0]
        image.save(out_path)
        print(f"Saved → {out_path}")

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
