#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Calculate optimal blocking parameters for Conv3D configurations.

Blocking parameters: (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)

Constraints:
- C_in_block: multiple of 32, divides C_in (after alignment to 32)
- C_out_block: multiple of 32, divides out_channels
- T_out_block, H_out_block, W_out_block: powers of 2, divide output dimensions
- num_patches_in_block = T_out_block * H_out_block * W_out_block <= 64
- If C_in_block == 128 or C_out_block == 128, then num_patches_in_block <= 32
"""

import json
import math


def _out_size(in_size, pad, stride, k):
    """Calculate output size for a dimension."""
    return (in_size + 2 * pad - k) // stride + 1


def get_divisors_multiple_32(n, min_val=32):
    """Get divisors of n that are multiples of 32."""
    divisors = []
    for i in range(min_val, n + 1, 32):
        if n % i == 0:
            divisors.append(i)
    return divisors


def get_power_of_2_divisors(n):
    """Get power of 2 divisors of n."""
    divisors = []
    i = 0
    while 2**i <= n:
        if n % (2**i) == 0:
            divisors.append(2**i)
        i += 1
    return divisors


def align_to_32(value):
    """Align value to multiple of 32."""
    return ((value + 31) // 32) * 32


def calculate_blocking(input_shape, out_channels, kernel_size, stride, padding):
    """Calculate appropriate blocking parameters."""
    N, C_in, D_in, H_in, W_in = input_shape
    K_T, K_H, K_W = kernel_size
    S_T, S_H, S_W = stride
    P_T, P_H, P_W = padding

    # Calculate output dimensions
    D_out = _out_size(D_in, P_T, S_T, K_T)
    H_out = _out_size(H_in, P_H, S_H, K_H)
    W_out = _out_size(W_in, P_W, S_W, K_W)

    # C_in is aligned to 32
    C_in_aligned = align_to_32(C_in)

    # Get possible block sizes
    C_in_blocks = get_divisors_multiple_32(C_in_aligned)
    C_out_blocks = get_divisors_multiple_32(out_channels)
    T_out_blocks = get_power_of_2_divisors(D_out)
    H_out_blocks = get_power_of_2_divisors(H_out)
    W_out_blocks = get_power_of_2_divisors(W_out)

    print(f"\nInput: {input_shape}, Out: {out_channels}, Kernel: {kernel_size}")
    print(f"Output dims: D={D_out}, H={H_out}, W={W_out}")
    print(f"C_in_aligned: {C_in_aligned}")
    print(f"C_in_blocks options: {C_in_blocks}")
    print(f"C_out_blocks options: {C_out_blocks}")
    print(f"T_out_blocks options: {T_out_blocks}")
    print(f"H_out_blocks options: {H_out_blocks}")
    print(f"W_out_blocks options: {W_out_blocks}")

    # Find valid blocking configurations
    MAX_PATCHES = 64
    MAX_PATCHES_WITH_128 = 32

    valid_configs = []

    for C_in_block in C_in_blocks:
        for C_out_block in C_out_blocks:
            for T_out_block in T_out_blocks:
                for H_out_block in H_out_blocks:
                    for W_out_block in W_out_blocks:
                        num_patches = T_out_block * H_out_block * W_out_block

                        # Check constraints
                        if num_patches > MAX_PATCHES:
                            continue
                        if (
                            C_in_block == 128 or C_out_block == 128
                        ) and num_patches > MAX_PATCHES_WITH_128:
                            continue

                        valid_configs.append(
                            {
                                "blocking": (
                                    C_in_block,
                                    C_out_block,
                                    T_out_block,
                                    H_out_block,
                                    W_out_block,
                                ),
                                "num_patches": num_patches,
                                "score": num_patches,  # Higher is generally better
                            }
                        )

    # Sort by score (prefer larger blocks for better performance)
    valid_configs.sort(key=lambda x: x["score"], reverse=True)

    return valid_configs, (D_out, H_out, W_out)


def suggest_blocking(input_shape, out_channels, kernel_size, stride, padding):
    """Suggest good blocking parameters."""
    configs, output_dims = calculate_blocking(
        input_shape, out_channels, kernel_size, stride, padding
    )

    if not configs:
        print("  ⚠️  No valid configurations found!")
        return None

    # Heuristic: prefer configurations with larger channel blocks and balanced spatial blocks
    # Priority: larger C_in_block and C_out_block, then balanced spatial blocks

    # Filter for larger channel blocks
    max_c_in = max(c["blocking"][0] for c in configs)
    max_c_out = max(c["blocking"][1] for c in configs)

    # Prefer 128 channel blocks if available, otherwise go lower
    preferred_c_in = 128 if 128 in [c["blocking"][0] for c in configs] else max_c_in
    preferred_c_out = 128 if 128 in [c["blocking"][1] for c in configs] else max_c_out

    filtered = [
        c
        for c in configs
        if c["blocking"][0] == preferred_c_in and c["blocking"][1] == preferred_c_out
    ]

    if not filtered:
        filtered = configs

    # Among filtered, prefer balanced spatial blocking and more patches
    best = filtered[0]  # Already sorted by num_patches

    print(f"\n  ✅ Suggested blocking: {best['blocking']}")
    print(f"     Patches per block: {best['num_patches']}")
    print(f"     Output dims: {output_dims}")

    return best["blocking"]


def main():
    # Load configurations
    with open("locations.json", "r") as f:
        convolutions = json.load(f)

    print("=" * 80)
    print("CALCULATING BLOCKING PARAMETERS FOR MOCHI DECODER CONVOLUTIONS")
    print("=" * 80)

    blockings = []

    for i, conv in enumerate(convolutions):
        # Parse shapes
        input_shape = tuple(map(int, conv["tensor_types"]["input"]["shape"].split("x")))
        weight_shape = tuple(
            map(int, conv["tensor_types"]["weight"]["shape"].split("x"))
        )

        out_channels = weight_shape[0]
        kernel_size = weight_shape[2:5]

        # Get attributes
        attrs = conv["attributes"]
        stride = tuple(map(int, attrs["window_strides"]))
        padding_list = list(map(int, attrs["padding"]))
        padding = (padding_list[0], padding_list[2], padding_list[4])  # T, H, W

        print(f"\n{'='*80}")
        print(f"Configuration {i+1}: {conv['location'][:60]}...")
        print(f"{'='*80}")

        blocking = suggest_blocking(
            input_shape, out_channels, kernel_size, stride, padding
        )
        blockings.append(blocking)

    print("\n" + "=" * 80)
    print("SUMMARY - BLOCKING PARAMETERS FOR TEST FILE")
    print("=" * 80)
    print()

    for i, blocking in enumerate(blockings):
        if blocking:
            print(f"    {blocking},  # conv{i+1}")
        else:
            print(f"    None,  # conv{i+1} - NO VALID BLOCKING FOUND")

    print("\n" + "=" * 80)
    print("REASONING")
    print("=" * 80)
    print(
        """
Blocking parameters are used to tile the computation for better performance:

1. **C_in_block, C_out_block**: Channel blocking
   - Must be multiples of 32 (alignment requirement)
   - Must divide the channel count
   - Prefer 128 when possible for better efficiency
   - If 128, spatial patches limited to 32

2. **T_out_block, H_out_block, W_out_block**: Spatial output blocking
   - Must be powers of 2
   - Must divide output dimensions
   - Product (num_patches) must be <= 64
   - Larger blocks generally better (more work per kernel launch)

3. **Strategy used**:
   - Maximize channel block sizes (prefer 128)
   - Then maximize spatial blocks within patch constraint
   - Balance between T, H, W blocks based on output dimensions
    """
    )


if __name__ == "__main__":
    main()
