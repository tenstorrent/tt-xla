# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end MLP-Mixer demo running through the D2M / TTMetal backend via XLA.

This script:
  1. Builds an MLP-Mixer model using D2M-friendly ops.
  2. Creates random image input and extracts patches on CPU.
  3. Runs the model on CPU to produce a golden reference.
  4. Runs the same model through XLA -> D2M -> TTMetal -> Flatbuffer -> Runtime.
  5. Compares device output against the CPU golden.

Usage:
    python examples/pytorch/d2m/mlp_mixer_e2e.py --dry-run
    python examples/pytorch/d2m/mlp_mixer_e2e.py
"""

import argparse
import os
import sys
import time

import pjrt_plugin_tt

# Select the TTMetal device runtime before any torch_xla import triggers plugin
# registration / device creation. Buffers must be created under the TTMetal
# runtime so their runtime tags match the flatbuffer produced by compilation.
pjrt_plugin_tt.set_backend("ttmetal")

import torch
from mlp_mixer import MIXER_TINY, build_mixer, create_patches

DTYPE = torch.bfloat16
ATOL = 0.10
RTOL = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="MLP-Mixer D2M E2E demo")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate model on CPU only"
    )
    parser.add_argument(
        "--export-ir",
        type=str,
        default=None,
        help="Directory to export intermediate MLIR artifacts",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Demo batch size")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Override the number of mixer blocks",
    )
    parser.add_argument(
        "--single-block",
        action="store_true",
        help="Run the smallest useful one-block bringup config",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Number of timed device iterations (after warmup)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations to amortize compile/cache",
    )
    return parser.parse_args()


def build_config(args):
    config = dict(MIXER_TINY)
    if args.num_blocks is not None:
        config["num_blocks"] = args.num_blocks
    if args.single_block:
        config["num_blocks"] = 1
    return config


def dry_run(args):
    config = build_config(args)
    model = build_mixer(config, dtype=DTYPE)
    model.eval()

    batch = args.batch_size
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    in_channels = config["in_channels"]
    num_patches = (image_size // patch_size) ** 2
    patch_dim = in_channels * patch_size * patch_size

    print(f"Config: {config}")
    print(f"Input patch tensor shape: [{batch}, {num_patches}, {patch_dim}]")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    images = torch.randn(batch, in_channels, image_size, image_size, dtype=DTYPE)
    patches = create_patches(images, patch_size)
    with torch.no_grad():
        output = model(patches)

    print(f"Output shape: {list(output.shape)}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print("\nD2M op mapping:")
    print("  nn.Linear      -> MatmulOp + AddOp")
    print("  nn.GELU        -> GeluOp")
    print("  nn.LayerNorm   -> MeanOp + SubtractOp + MultiplyOp + RsqrtOp + AddOp")
    print("  transpose      -> PermuteOp")
    print("  mean(dim=1)    -> MeanOp")
    print("  residual add   -> AddOp")
    print("\nDRY RUN COMPLETE")
    return True


def hardware_run(args):
    import torch_xla
    import torch_xla.core.xla_model as xm

    config = build_config(args)
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    in_channels = config["in_channels"]

    compile_opts = {"backend": "ttmetal_flatbuffer"}
    if args.export_ir:
        os.makedirs(args.export_ir, exist_ok=True)
        compile_opts["export_path"] = args.export_ir
        print(f"IR export enabled -> {args.export_ir}")
    torch_xla.set_custom_compile_options(compile_opts)

    device = xm.xla_device()
    print(f"XLA device: {device}")
    print(f"Config: {config}")

    model = build_mixer(config, dtype=DTYPE)
    model.eval()

    batch = args.batch_size
    images = torch.randn(batch, in_channels, image_size, image_size, dtype=DTYPE)
    patches = create_patches(images, patch_size)

    with torch.no_grad():
        cpu_start = time.perf_counter()
        golden = model(patches)
        cpu_time = time.perf_counter() - cpu_start

    print(f"CPU time: {cpu_time * 1000:.1f} ms")
    print(f"Patches shape: {list(patches.shape)}")

    model_dev = model.to(device)
    patches_dev = patches.to(device)
    xm.wait_device_ops()

    with torch.no_grad():
        e2e_start = time.perf_counter()
        out_dev = model_dev(patches_dev)
        xm.mark_step()
        xm.wait_device_ops()
        e2e_time = time.perf_counter() - e2e_start

        for _ in range(max(0, args.warmup - 1)):
            out_dev = model_dev(patches_dev)
            xm.mark_step()
        xm.wait_device_ops()

        per_iter = []
        for _ in range(args.iters):
            iter_start = time.perf_counter()
            out_dev = model_dev(patches_dev)
            xm.mark_step()
            xm.wait_device_ops()
            per_iter.append(time.perf_counter() - iter_start)

        out = out_dev.to("cpu")

    device_time = sum(per_iter) / len(per_iter) if per_iter else 0.0
    device_min = min(per_iter) if per_iter else 0.0

    golden_f32 = golden.float()
    out_f32 = out.float()
    abs_diff = (out_f32 - golden_f32).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    g_flat = golden_f32.flatten()
    o_flat = out_f32.flatten()
    g_centered = g_flat - g_flat.mean()
    o_centered = o_flat - o_flat.mean()
    pcc = (
        (g_centered * o_centered).sum() / (g_centered.norm() * o_centered.norm() + 1e-8)
    ).item()

    print(f"Device time (first run, incl. compile): {e2e_time * 1000:.1f} ms")
    print(
        f"Device time (mean over {len(per_iter)} iters): {device_time * 1000:.2f} ms"
        f" | min: {device_min * 1000:.2f} ms"
    )
    print(f"Max absolute diff: {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")
    print(f"PCC: {pcc:.6f}")

    passed = bool(
        torch.allclose(out_f32, golden_f32, atol=ATOL, rtol=RTOL) and pcc > 0.90
    )
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    torch.manual_seed(0)
    args = parse_args()
    ok = dry_run(args) if args.dry_run else hardware_run(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
