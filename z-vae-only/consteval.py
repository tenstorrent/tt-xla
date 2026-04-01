# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Deduplicated const-eval helper functions for the VAE decoder.

The codegen main.py has ~147 const-eval functions but only ~9 unique patterns.
This module provides clean, parameterized versions of each pattern.

All functions operate on TTNN tensors and are intended to be called during
weight preparation (before the forward pass), not during inference.
"""

import ttnn

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
)


# ---------------------------------------------------------------------------
# GroupNorm weight preparation
# ---------------------------------------------------------------------------


def prepare_norm_weight(tensor, channels, num_groups, device):
    """Prepare GroupNorm gamma/beta weight: 1D [C] -> [1, G, C/G, 1] in FLOAT32 on device.

    Pattern from const_eval_1 / const_eval_2 / etc.:
        to_device -> TILE -> reshape [1,C,1,1] -> permute NHWC -> typecast FLOAT32
        -> permute NCHW -> reshape [1, G, C/G, 1]

    Used for all ResNet block norm1/norm2 weights and biases, and conv_norm_out.
    """
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.reshape(t, [1, channels, 1, 1], memory_config=DRAM)
    t = ttnn.permute(t, [0, 2, 3, 1], memory_config=DRAM, pad_value=0.0)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM)
    t = ttnn.permute(t, [0, 3, 1, 2], memory_config=DRAM, pad_value=0.0)
    t = ttnn.reshape(
        t, [1, num_groups, channels // num_groups, 1], memory_config=DRAM
    )
    return t


def prepare_attn_norm_weight(tensor, channels, num_groups, device):
    """Prepare attention GroupNorm weight: 1D [C] -> [1, G, C/G, 1] FLOAT32 on device.

    Pattern from const_eval_24 / const_eval_25 (slightly different reshape path):
        to_device -> TILE -> reshape [1,C,1] -> permute [0,2,1] -> reshape [1,C]
        -> typecast FLOAT32 -> reshape [1,1,C] -> permute [0,2,1]
        -> reshape [1,G,C/G,1]

    Used for attention group_norm weight and bias.
    """
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.reshape(t, [1, channels, 1], memory_config=DRAM)
    t = ttnn.permute(t, [0, 2, 1], memory_config=DRAM, pad_value=0.0)
    t = ttnn.reshape(t, [1, channels], memory_config=DRAM)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM)
    t = ttnn.reshape(t, [1, 1, channels], memory_config=DRAM)
    t = ttnn.permute(t, [0, 2, 1], memory_config=DRAM, pad_value=0.0)
    t = ttnn.reshape(
        t, [1, num_groups, channels // num_groups, 1], memory_config=DRAM
    )
    return t


# ---------------------------------------------------------------------------
# Attention bias preparation
# ---------------------------------------------------------------------------


def prepare_attn_bias(tensor, channels, seq_len, device):
    """Prepare attention bias: 1D [C] -> [1, seq_len, C] via repeat.

    Pattern from const_eval_26..29:
        to_device -> TILE -> reshape [1,1,C] -> repeat [1, seq_len, 1]

    Used for to_q.bias, to_k.bias, to_v.bias, to_out.0.bias in the attention block.
    """
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.reshape(t, [1, 1, channels], memory_config=DRAM)
    t = ttnn.repeat(t, ttnn.Shape([1, seq_len, 1]), memory_config=DRAM)
    return t


# ---------------------------------------------------------------------------
# Conv2d weight and bias preparation
# ---------------------------------------------------------------------------


def prepare_conv_weight(
    tensor,
    device,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    has_bias,
    act_block_h_override=0,
    slice_config=None,
):
    """Prepare conv2d weight via ttnn.prepare_conv_weights.

    Pattern from const_eval_3..147 (conv weight variants):
        Calls ttnn.prepare_conv_weights with all conv parameters.
    """
    kwargs = dict(
        weight_tensor=tensor,
        input_memory_config=DRAM,
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        has_bias=has_bias,
        groups=groups,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=act_block_h_override,
            enable_kernel_stride_folding=False,
        ),
        compute_config=COMPUTE_CONFIG,
    )
    if slice_config is not None:
        kwargs["slice_config"] = slice_config
    return ttnn.prepare_conv_weights(**kwargs)


def prepare_conv_bias(
    tensor,
    device,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    act_block_h_override=0,
):
    """Prepare conv2d bias via ttnn.prepare_conv_bias.

    Pattern from const_eval_4..148 (conv bias variants):
        Reshape 1D bias -> [1, C, 1, 1] -> permute NHWC -> ROW_MAJOR -> from_device
        -> ttnn.prepare_conv_bias(...)
    """
    channels = out_channels
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.reshape(t, [1, channels, 1, 1], memory_config=DRAM)
    t = ttnn.permute(t, [0, 2, 3, 1], memory_config=DRAM, pad_value=0.0)
    t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    t = ttnn.from_device(t)
    return ttnn.prepare_conv_bias(
        bias_tensor=t,
        input_memory_config=DRAM,
        input_layout=ttnn.Layout.TILE,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=act_block_h_override,
            enable_kernel_stride_folding=False,
        ),
        compute_config=COMPUTE_CONFIG,
    )


# ---------------------------------------------------------------------------
# Tensor creation helpers
# ---------------------------------------------------------------------------


def create_ones_tensor(shape, device):
    """Create a tensor filled with 1.0 in BFLOAT16 TILE layout on device.

    Pattern from const_eval_30 / const_eval_31 etc.
    """
    return ttnn.full(
        shape=ttnn.Shape(shape),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )


def create_ones_permuted(shape, device):
    """Create a ones tensor and also return a permuted version [0,1,3,2].

    Used for upsample interpolation steps.

    Returns:
        (ones, ones_permuted) tuple.
    """
    ones = create_ones_tensor(shape, device)
    ones_perm = ttnn.permute(ones, [0, 1, 3, 2], memory_config=DRAM)
    return ones, ones_perm


# ---------------------------------------------------------------------------
# Upsample mask creation
# ---------------------------------------------------------------------------


def create_upsample_mask(src_size, dst_size, device):
    """Create nearest-neighbor upsample indicator matrix [src_size, dst_size].

    M[j,i] = 1.0 if floor(i * src_size / dst_size) == j, else 0.0.
    This matrix is used as a right-hand matmul weight: [N, src_size] @ [src_size, dst_size] -> [N, dst_size].

    Pattern from const_eval_0:
        - Create arange [src_size] as INT32 -> reshape [1,src] -> permute [src,1]
        - Create arange with scale step [dst_size] as FLOAT32 -> floor -> INT32
          -> reshape [dst,1] -> permute [1,dst]
        - eq([1,dst], [src,1]) -> broadcast to [src, dst]
    """
    # Source indices [src_size]
    src_values = list(range(src_size))
    src_t = ttnn.Tensor(
        src_values,
        [src_size],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM,
    )

    # Destination indices with scale step: [0.0, 0.5, 1.0, 1.5, ...] for 2x upscale
    scale = src_size / dst_size
    dst_values = [i * scale for i in range(dst_size)]
    dst_t = ttnn.Tensor(
        dst_values,
        [dst_size],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM,
    )

    # Floor to get integer source indices
    dst_floored = ttnn.floor(dst_t, memory_config=DRAM)
    dst_int = ttnn.typecast(dst_floored, ttnn.DataType.INT32, memory_config=DRAM)

    # Reshape for broadcast comparison (matching codegen const_eval_0 exactly):
    # dst: [dst_size] -> reshape [dst_size, 1] -> permute [1, dst_size] = row vector
    dst_col = ttnn.reshape(dst_int, [dst_size, 1], memory_config=DRAM)
    dst_row = ttnn.permute(dst_col, [1, 0], memory_config=DRAM, pad_value=0.0)

    # src: [src_size] -> reshape [1, src_size] -> permute [src_size, 1] = column vector
    src_row = ttnn.reshape(src_t, [1, src_size], memory_config=DRAM)
    src_col = ttnn.permute(src_row, [1, 0], memory_config=DRAM, pad_value=0.0)

    # Broadcast eq: [1, dst_size] == [src_size, 1] -> [src_size, dst_size]
    mask = ttnn.eq(dst_row, src_col, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
    return mask


def create_scalar_tensor(value, device):
    """Create a scalar BFLOAT16 TILE tensor [1,1,1,1] on device."""
    return ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=value,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )
