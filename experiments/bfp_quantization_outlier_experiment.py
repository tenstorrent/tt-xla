# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BFP Quantization Outlier Experiment

Demonstrates how outlier values affect accuracy in block floating point formats
(bfloat8_b and bfloat4_b) on Tenstorrent hardware.

Block formats split tensors into blocks of 16 consecutive floats that share
a single 8-bit exponent. Outliers push the shared exponent up, reducing
precision for smaller values in the same block.

Usage:
    python experiments/bfp_quantization_outlier_experiment.py --outlier-magnitude 5
    python experiments/bfp_quantization_outlier_experiment.py --outlier-magnitude 20
"""

import argparse

import torch
import ttnn


def create_base_tensor(rows, cols, seed=42):
    """Create a uniform random bfloat16 tensor (no outliers)."""
    torch.manual_seed(seed)
    return torch.rand(rows, cols, dtype=torch.bfloat16)


def inject_outliers(tensor, outlier_magnitude):
    """Return a copy with outliers at positions 8 and 24 in each row."""
    out = tensor.clone()
    for i in range(out.shape[0]):
        out[i, 8] *= outlier_magnitude
        out[i, 24] *= outlier_magnitude
    return out


def build_outlier_mask(rows, cols):
    """Return a boolean mask that is True at outlier positions."""
    mask = torch.zeros(rows, cols, dtype=torch.bool)
    for i in range(rows):
        mask[i, 8] = True
        mask[i, 24] = True
    return mask


def quantize_via_ttnn(original_torch, dtype, device):
    """Send tensor to device, typecast to dtype, return as torch tensor."""
    # Convert to ttnn tensor on device in TILE layout (required for bfp types)
    tt_tensor = ttnn.from_torch(
        original_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Typecast to target block format
    tt_quantized = ttnn.typecast(tt_tensor, dtype)

    # Cast back to bfloat16 so we can bring it to host and compare
    tt_dequantized = ttnn.typecast(tt_quantized, ttnn.bfloat16)

    # Move to host and convert to torch
    result = ttnn.to_torch(tt_dequantized)

    # Trim any tile padding back to original shape
    result = result[: original_torch.shape[0], : original_torch.shape[1]]

    return result


def compute_errors(original, quantized, outlier_mask=None):
    """Compute MAE (overall/outlier/non-outlier) and relative L2 error."""
    orig_f32 = original.float()
    quant_f32 = quantized.float()
    abs_diff = (orig_f32 - quant_f32).abs()

    overall = abs_diff.mean().item()
    if outlier_mask is not None and outlier_mask.any():
        outlier_err = abs_diff[outlier_mask].mean().item()
        non_outlier_err = abs_diff[~outlier_mask].mean().item()
    else:
        outlier_err = 0.0
        non_outlier_err = overall

    # Relative L2 error: ||x - x_hat||_2 / ||x||_2
    diff_norm = torch.linalg.vector_norm(orig_f32 - quant_f32).item()
    orig_norm = torch.linalg.vector_norm(orig_f32).item()
    rel_l2 = diff_norm / max(orig_norm, 1e-12)

    return overall, outlier_err, non_outlier_err, rel_l2


def main():
    parser = argparse.ArgumentParser(description="BFP quantization outlier experiment")
    parser.add_argument(
        "--outlier-magnitude",
        type=float,
        default=5.0,
        help="Multiplier for outlier elements (default: 5.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    rows, cols = 32, 32

    # Create base tensor and variant with outliers (same random seed)
    base = create_base_tensor(rows, cols, args.seed)
    with_outliers = inject_outliers(base, args.outlier_magnitude)
    outlier_mask = build_outlier_mask(rows, cols)

    print(f"Tensor shape: {base.shape}")
    print(f"Tensor dtype: {base.dtype}")
    print(f"Outlier magnitude multiplier: {args.outlier_magnitude}")
    print(f"Number of outlier positions: {outlier_mask.sum().item()}")
    print()
    print(f"{'':>30} {'No outliers':>14} {'With outliers':>14}")
    print(
        f"{'Value range':<30} [{base.min().item():.4f}, {base.max().item():.4f}] [{with_outliers.min().item():.4f}, {with_outliers.max().item():.4f}]"
    )
    print(
        f"{'Mean (all)':<30} {base.float().mean().item():>14.4f} {with_outliers.float().mean().item():>14.4f}"
    )
    print(
        f"{'Mean (non-outlier positions)':<30} {base[~outlier_mask].float().mean().item():>14.4f} {with_outliers[~outlier_mask].float().mean().item():>14.4f}"
    )
    print(
        f"{'Mean (outlier positions)':<30} {base[outlier_mask].float().mean().item():>14.4f} {with_outliers[outlier_mask].float().mean().item():>14.4f}"
    )
    print()

    # Open TT device
    device = ttnn.open_device(device_id=0)
    print("opened TT device")
    print()

    try:
        # Quantize both tensors through both formats
        base_bfp8 = quantize_via_ttnn(base, ttnn.bfloat8_b, device)
        base_bfp4 = quantize_via_ttnn(base, ttnn.bfloat4_b, device)
        out_bfp8 = quantize_via_ttnn(with_outliers, ttnn.bfloat8_b, device)
        out_bfp4 = quantize_via_ttnn(with_outliers, ttnn.bfloat4_b, device)

        # Compute errors — no outlier mask for baseline, with mask for outlier variant
        b8_overall, _, _, b8_l2 = compute_errors(base, base_bfp8)
        b4_overall, _, _, b4_l2 = compute_errors(base, base_bfp4)
        o8_overall, o8_out, o8_non, o8_l2 = compute_errors(
            with_outliers, out_bfp8, outlier_mask
        )
        o4_overall, o4_out, o4_non, o4_l2 = compute_errors(
            with_outliers, out_bfp4, outlier_mask
        )

        # ── Side-by-side summary ──
        w = 78
        print("=" * w)
        print("SIDE-BY-SIDE: NO OUTLIERS vs WITH OUTLIERS")
        print("=" * w)

        hdr = f"{'Metric':<30} {'No outliers':>22} {'With outliers':>22}"
        sub = f"{'':30} {'BFP8':>10} {'BFP4':>10}   {'BFP8':>10} {'BFP4':>10}"
        print(hdr)
        print(sub)
        print("-" * w)

        print(
            f"{'Overall MAE':<30} {b8_overall:>10.6f} {b4_overall:>10.6f}   {o8_overall:>10.6f} {o4_overall:>10.6f}"
        )
        print(
            f"{'Outlier MAE':<30} {'—':>10} {'—':>10}   {o8_out:>10.6f} {o4_out:>10.6f}"
        )
        print(
            f"{'Non-outlier MAE':<30} {'—':>10} {'—':>10}   {o8_non:>10.6f} {o4_non:>10.6f}"
        )
        print(
            f"{'Relative L2 (||x-x̂||/||x||)':<30} {b8_l2:>10.6f} {b4_l2:>10.6f}   {o8_l2:>10.6f} {o4_l2:>10.6f}"
        )
        print()

        # ── Ratio tables ──
        print("=" * w)
        print("ERROR RATIOS")
        print("=" * w)
        print(f"{'Ratio':<40} {'BFP8':>16} {'BFP4':>16}")
        print("-" * w)
        print(
            f"{'BFP4/BFP8 (no outliers)':<40} {'':>16} {b4_overall / max(b8_overall, 1e-12):>15.2f}x"
        )
        print(
            f"{'BFP4/BFP8 (with outliers)':<40} {'':>16} {o4_overall / max(o8_overall, 1e-12):>15.2f}x"
        )
        print(
            f"{'Outlier/No-outlier MAE':<40} {o8_overall / max(b8_overall, 1e-12):>15.2f}x {o4_overall / max(b4_overall, 1e-12):>15.2f}x"
        )
        print(
            f"{'Outlier/No-outlier Relative L2':<40} {o8_l2 / max(b8_l2, 1e-12):>15.2f}x {o4_l2 / max(b4_l2, 1e-12):>15.2f}x"
        )
        print()

        # ── Sample block ──
        print("Sample block (row 0, elements 0-15):")
        print(f"  Base:            {base[0, :16].tolist()}")
        print(f"  Base BFP8:       {base_bfp8[0, :16].tolist()}")
        print(f"  Base BFP4:       {base_bfp4[0, :16].tolist()}")
        print(f"  With outliers:   {with_outliers[0, :16].tolist()}")
        print(f"  Outlier BFP8:    {out_bfp8[0, :16].tolist()}")
        print(f"  Outlier BFP4:    {out_bfp4[0, :16].tolist()}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
