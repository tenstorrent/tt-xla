# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Clean TTNN implementation of the SD3.5 VAE decoder.

Self-contained -- no imports from sibling directories.
Extracted from the 22K-line codegen main.py and organized into clean
LightweightModule classes matching the PyTorch reference model_pt.py.

Architecture:
    VaeDecoderTTNN
    +-- conv_in: Conv2d(16, 512)
    +-- mid_block: MidBlock
    |   +-- resnet0: ResNetBlock(512, 512)
    |   +-- attention: AttentionBlock(512)
    |   +-- resnet1: ResNetBlock(512, 512)
    +-- up_blocks[0]: UpBlock(512, 512, upsample=True)    # 160x90 -> 320x180
    |   +-- resnets[0,1,2]: ResNetBlock(512, 512)
    |   +-- upsample: Upsample(512, 320, 180)
    +-- up_blocks[1]: UpBlock(512, 512, upsample=True)    # 320x180 -> 640x360
    |   +-- resnets[0,1,2]: ResNetBlock(512, 512)
    |   +-- upsample: Upsample(512, 640, 360)
    +-- up_blocks[2]: UpBlock(512, 256, upsample=True)    # 640x360 -> 1280x720
    |   +-- resnets[0]: ResNetBlock(512, 256, conv_shortcut=True)
    |   +-- resnets[1,2]: ResNetBlock(256, 256)
    |   +-- upsample: Upsample(256, 1280, 720)
    +-- up_blocks[3]: UpBlock(256, 128, upsample=False)   # 1280x720
    |   +-- resnets[0]: ResNetBlock(256, 128, conv_shortcut=True)
    |   +-- resnets[1,2]: ResNetBlock(128, 128)
    +-- conv_norm_out: GroupNorm(32, 128)
    +-- conv_act: SiLU
    +-- conv_out: Conv2d(128, 3)
"""

import json
import os

import torch
import ttnn

from consteval import (
    COMPUTE_CONFIG,
    DRAM,
    create_upsample_mask,
    prepare_attn_bias,
    prepare_attn_norm_weight,
    prepare_conv_bias,
    prepare_conv_weight,
    prepare_norm_weight,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_GROUPS = 32
GN_EPS = 1e-6  # GroupNorm epsilon
ATTN_SCALE = 1.0 / 512**0.5  # 1/sqrt(head_dim) for single-head attention
SDPA_SCALE = ATTN_SCALE

# VAE scaling factors (SD3.5)
SCALING_FACTOR = 1.5305
SHIFT_FACTOR = 0.0609


# ---------------------------------------------------------------------------
# LightweightModule base class
# ---------------------------------------------------------------------------


class LightweightModule:
    """Base class for TTNN modules. Stores weights as attributes, no nn.Module."""

    def __init__(self):
        pass


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def group_norm(x, gamma_ce, beta_ce, channels, height, width, device):
    """Manual GroupNorm: x is [1, C, H, W] NCHW.

    Matches the codegen pattern exactly:
        reshape -> mean -> subtract -> square -> mean -> add eps -> rsqrt
        -> multiply -> scale -> shift -> reshape back

    gamma_ce and beta_ce are const-eval prepared: [1, G, C/G, 1] FLOAT32.
    """
    hw = height * width
    x = ttnn.reshape(x, [1, NUM_GROUPS, channels // NUM_GROUPS, hw], memory_config=DRAM)

    # Mean over spatial + channel-group dims
    mean = ttnn.mean(
        x, [2, 3], True, memory_config=DRAM, compute_kernel_config=COMPUTE_CONFIG
    )
    neg_mean = ttnn.neg(mean, memory_config=DRAM)
    ttnn.deallocate(mean, False)
    centered = ttnn.add(x, neg_mean, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(neg_mean, False)
    ttnn.deallocate(x, False)

    # Variance
    var = ttnn.multiply(
        centered, centered, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM
    )
    var_mean = ttnn.mean(
        var, [2, 3], True, memory_config=DRAM, compute_kernel_config=COMPUTE_CONFIG
    )
    ttnn.deallocate(var, False)

    # eps_tensor: scalar 1e-6
    eps_t = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=GN_EPS,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )
    var_eps = ttnn.add(
        var_mean, eps_t, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM
    )
    ttnn.deallocate(var_mean, False)
    inv_std = ttnn.rsqrt(var_eps, fast_and_approximate_mode=False, memory_config=DRAM)
    ttnn.deallocate(var_eps, False)

    # Normalize, scale, shift
    normed = ttnn.multiply(
        centered, inv_std, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM
    )
    ttnn.deallocate(inv_std, False)
    ttnn.deallocate(centered, False)
    scaled = ttnn.multiply(
        normed, gamma_ce, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM
    )
    ttnn.deallocate(normed, False)
    result = ttnn.add(
        scaled, beta_ce, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM
    )
    ttnn.deallocate(scaled, False)

    # Reshape back to [1, C, H, W]
    result = ttnn.reshape(result, [1, channels, height, width], memory_config=DRAM)
    return result


def conv2d(
    x,
    weight,
    bias,
    device,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size=None,
    stride=None,
    padding=None,
    act_block_h_override=0,
    slice_config=None,
):
    """Conv2d matching the codegen pattern.

    Input x: [1, C_in, H, W] NCHW.
    Permute to NHWC -> reshape to [1, 1, H*W, C] -> ttnn.conv2d -> reshape back.

    Weight and bias must already be prepared via prepare_conv_weight / prepare_conv_bias.
    """
    if kernel_size is None:
        kernel_size = [3, 3]
    if stride is None:
        stride = [1, 1]
    if padding is None:
        padding = [1, 1, 1, 1]
    if slice_config is None:
        # Use DRAMSliceWidth for large spatial dims with 3x3 kernels (matching codegen)
        # 1x1 conv_shortcut and 160x90 convs use L1Full
        if height * width > 160 * 90 and kernel_size == [3, 3]:
            slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
            )
        else:
            slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dL1Full, num_slices=0
            )

    hw = height * width

    # NCHW -> NHWC -> [1, 1, H*W, C_in]
    x = ttnn.permute(x, [0, 2, 3, 1], memory_config=DRAM, pad_value=0.0)
    x = ttnn.reshape(x, [1, 1, hw, in_channels], memory_config=DRAM)

    x = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=1,
        input_height=height,
        input_width=width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=[1, 1],
        groups=1,
        bias_tensor=bias,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=act_block_h_override,
            enable_kernel_stride_folding=False,
        ),
        compute_config=COMPUTE_CONFIG,
        slice_config=slice_config,
        memory_config=DRAM,
    )

    # Output is [1, H_out*W_out, out_C] -> reshape to NHWC -> permute to NCHW
    # For same-padding convs (stride=1, padding=1): H_out=H, W_out=W
    out_h = height // stride[0]
    out_w = width // stride[1]
    x = ttnn.reshape(x, [1, out_h, out_w, out_channels], memory_config=DRAM)
    x = ttnn.permute(x, [0, 3, 1, 2], memory_config=DRAM, pad_value=0.0)
    return x


# ---------------------------------------------------------------------------
# ResNetBlock
# ---------------------------------------------------------------------------


class ResNetBlock(LightweightModule):
    """Residual block: GroupNorm1 + SiLU + Conv1 + GroupNorm2 + SiLU + Conv2.

    Optionally has a 1x1 conv_shortcut when in_channels != out_channels.
    Residual connection: (x + residual) / 2 (output_scale_factor=2).

    Weight name prefix example: "decoder.mid_block.resnets.0"
    """

    def __init__(
        self,
        norm1_weight_ce,
        norm1_bias_ce,
        conv1_weight,
        conv1_bias,
        norm2_weight_ce,
        norm2_bias_ce,
        conv2_weight,
        conv2_bias,
        in_channels,
        out_channels,
        height,
        width,
        act_block_h_override=1024,
        conv_shortcut_weight=None,
        conv_shortcut_bias=None,
        conv_shortcut_act_block_h=0,
        device=None,
    ):
        super().__init__()
        self.norm1_weight_ce = norm1_weight_ce
        self.norm1_bias_ce = norm1_bias_ce
        self.conv1_weight = conv1_weight
        self.conv1_bias = conv1_bias
        self.norm2_weight_ce = norm2_weight_ce
        self.norm2_bias_ce = norm2_bias_ce
        self.conv2_weight = conv2_weight
        self.conv2_bias = conv2_bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.act_block_h_override = act_block_h_override
        self.conv_shortcut_weight = conv_shortcut_weight
        self.conv_shortcut_bias = conv_shortcut_bias
        self.conv_shortcut_act_block_h = conv_shortcut_act_block_h
        self.has_shortcut = conv_shortcut_weight is not None
        self.device = device

    def forward(self, x):
        residual = x

        # Typecast to FLOAT32 before GroupNorm (matching codegen)
        x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM)

        # GroupNorm1 + SiLU + Conv1
        x = group_norm(
            x,
            self.norm1_weight_ce,
            self.norm1_bias_ce,
            self.in_channels,
            self.height,
            self.width,
            self.device,
        )
        prev = x
        x = ttnn.silu(x, memory_config=DRAM)
        ttnn.deallocate(prev, False)
        prev = x
        x = ttnn.typecast(x, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(prev, False)
        x = conv2d(
            x,
            self.conv1_weight,
            self.conv1_bias,
            self.device,
            self.in_channels,
            self.out_channels,
            self.height,
            self.width,
            act_block_h_override=self.act_block_h_override,
        )

        # GroupNorm2 + SiLU + Conv2
        prev = x
        x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(prev, False)
        x = group_norm(
            x,
            self.norm2_weight_ce,
            self.norm2_bias_ce,
            self.out_channels,
            self.height,
            self.width,
            self.device,
        )
        prev = x
        x = ttnn.silu(x, memory_config=DRAM)
        ttnn.deallocate(prev, False)
        prev = x
        x = ttnn.typecast(x, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(prev, False)
        x = conv2d(
            x,
            self.conv2_weight,
            self.conv2_bias,
            self.device,
            self.out_channels,
            self.out_channels,
            self.height,
            self.width,
            act_block_h_override=self.act_block_h_override,
        )

        # Optional conv_shortcut on residual
        if self.has_shortcut:
            residual = conv2d(
                residual,
                self.conv_shortcut_weight,
                self.conv_shortcut_bias,
                self.device,
                self.in_channels,
                self.out_channels,
                self.height,
                self.width,
                kernel_size=[1, 1],
                padding=[0, 0, 0, 0],
                act_block_h_override=self.conv_shortcut_act_block_h,
            )

        # Residual add (output_scale_factor=1, no division)
        x = ttnn.add(x, residual, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        return x


# ---------------------------------------------------------------------------
# AttentionBlock
# ---------------------------------------------------------------------------


class AttentionBlock(LightweightModule):
    """Self-attention block used in the VAE mid_block.

    Single head with head_dim = channels (512).
    Codegen pattern: GroupNorm -> Q/K/V projections (matmul + bias) -> SDPA -> out projection.
    Residual: (out + x/2) ... actually (out + residual) / 2 per codegen.

    Weight name prefix: "decoder.mid_block.attentions.0"
    """

    def __init__(
        self,
        group_norm_weight_ce,
        group_norm_bias_ce,
        q_weight,
        q_bias_ce,
        k_weight,
        k_bias_ce,
        v_weight,
        v_bias_ce,
        out_weight,
        out_bias_ce,
        channels,
        height,
        width,
        device,
    ):
        super().__init__()
        self.group_norm_weight_ce = group_norm_weight_ce
        self.group_norm_bias_ce = group_norm_bias_ce
        self.q_weight = q_weight
        self.q_bias_ce = q_bias_ce
        self.k_weight = k_weight
        self.k_bias_ce = k_bias_ce
        self.v_weight = v_weight
        self.v_bias_ce = v_bias_ce
        self.out_weight = out_weight
        self.out_bias_ce = out_bias_ce
        self.channels = channels
        self.height = height
        self.width = width
        self.device = device

    def forward(self, x):
        residual = x
        seq_len = self.height * self.width  # H*W

        # Typecast to FLOAT32 before GroupNorm (matching codegen)
        x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM)

        # GroupNorm
        x_normed = group_norm(
            x,
            self.group_norm_weight_ce,
            self.group_norm_bias_ce,
            self.channels,
            self.height,
            self.width,
            self.device,
        )
        ttnn.deallocate(x, False)

        # Reshape to sequence: [1, C, H, W] -> NHWC -> [1, 1, H*W, C]
        x_seq = ttnn.permute(
            x_normed, [0, 2, 3, 1], memory_config=DRAM, pad_value=0.0
        )
        ttnn.deallocate(x_normed, False)
        x_seq = ttnn.reshape(
            x_seq, [1, 1, seq_len, self.channels], memory_config=DRAM
        )

        # Q projection
        q = ttnn.matmul(
            x_seq, self.q_weight, transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16, compute_kernel_config=COMPUTE_CONFIG,
        )
        q = ttnn.add(q, self.q_bias_ce, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        q = ttnn.typecast(q, ttnn.DataType.FLOAT32, memory_config=DRAM)
        q = ttnn.reshape(q, [1, 1, seq_len, self.channels], memory_config=DRAM)

        # K projection
        k = ttnn.matmul(
            x_seq, self.k_weight, transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16, compute_kernel_config=COMPUTE_CONFIG,
        )
        k = ttnn.add(k, self.k_bias_ce, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        k = ttnn.typecast(k, ttnn.DataType.FLOAT32, memory_config=DRAM)
        k = ttnn.reshape(k, [1, 1, seq_len, self.channels], memory_config=DRAM)

        # V projection
        v = ttnn.matmul(
            x_seq, self.v_weight, transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16, compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(x_seq, False)
        v = ttnn.add(v, self.v_bias_ce, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        v = ttnn.typecast(v, ttnn.DataType.FLOAT32, memory_config=DRAM)
        v = ttnn.reshape(v, [1, 1, seq_len, self.channels], memory_config=DRAM)

        # Cast to BFLOAT16 for SDPA
        q = ttnn.typecast(q, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        k = ttnn.typecast(k, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        v = ttnn.typecast(v, ttnn.DataType.BFLOAT16, memory_config=DRAM)

        # Scaled dot-product attention
        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False,
            scale=SDPA_SCALE, sliding_window_size=None, memory_config=DRAM,
        )
        ttnn.deallocate(q, False)
        ttnn.deallocate(k, False)
        ttnn.deallocate(v, False)

        # Output projection
        out = ttnn.reshape(out, [seq_len, self.channels], memory_config=DRAM)
        out = ttnn.matmul(
            out, self.out_weight, transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16, compute_kernel_config=COMPUTE_CONFIG,
        )
        out = ttnn.add(
            out, self.out_bias_ce, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM
        )

        # Reshape back to NCHW
        out = ttnn.reshape(
            out, [1, self.height, self.width, self.channels], memory_config=DRAM
        )
        out = ttnn.permute(out, [0, 3, 1, 2], memory_config=DRAM, pad_value=0.0)

        # Residual add (output_scale_factor=1, no division)
        out = ttnn.add(out, residual, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        return out


# ---------------------------------------------------------------------------
# Upsample
# ---------------------------------------------------------------------------


class Upsample(LightweightModule):
    """Nearest-neighbor 2x upsample via matmul with indicator matrices, then conv.

    The codegen implements nearest-neighbor upsampling as:
    1. Reshape x to flatten C and W dims: [C*W, H]
    2. Matmul with H-upsample mask [H, 2*H]^T -> [C*W, 2*H]
    3. Reshape/permute to get [1, C, 2*H, W]
    4. Reshape to flatten C and 2*H dims: [C*2*H, W]
    5. Matmul with W-upsample mask [W, 2*W]^T -> [C*2*H, 2*W]
    6. Reshape to [1, C, 2*H, 2*W]
    7. Apply 3x3 conv

    Weight name prefix example: "decoder.up_blocks.0.upsamplers.0"
    """

    def __init__(
        self,
        conv_weight,
        conv_bias,
        channels,
        in_height,
        in_width,
        upsample_mask_h,
        upsample_mask_w,
        act_block_h_override=1024,
        device=None,
    ):
        super().__init__()
        self.conv_weight = conv_weight
        self.conv_bias = conv_bias
        self.channels = channels
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = in_height * 2
        self.out_width = in_width * 2
        self.upsample_mask_h = upsample_mask_h
        self.upsample_mask_w = upsample_mask_w
        self.act_block_h_override = act_block_h_override
        self.device = device

    def forward(self, x):
        C = self.channels
        H = self.in_height
        W = self.in_width
        H2 = self.out_height
        W2 = self.out_width

        # The codegen transposes from NCHW [1,C,H,W] to NCWH [1,C,W,H] before
        # the upsample matmul (via broadcast-divide with a [1,C,W,H] ones tensor).
        # We do this explicitly with a permute.
        x = ttnn.permute(x, [0, 1, 3, 2], memory_config=DRAM, pad_value=0.0)
        # x is now [1, C, W, H]

        # Step 1: Upsample H dimension (H -> 2*H)
        # x: [1, C, W, H] -> [C*W, H]
        x = ttnn.reshape(x, [C * W, H], memory_config=DRAM)
        # matmul with mask [H, H2] transposed -> output [C*W, H2]
        x = ttnn.matmul(
            x,
            self.upsample_mask_h,
            memory_config=DRAM,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        # Reshape to [1, C, W, H2] then permute to [1, C, H2, W]
        x = ttnn.reshape(x, [1, C, W, H2], memory_config=DRAM)
        x = ttnn.permute(x, [0, 1, 3, 2], memory_config=DRAM, pad_value=0.0)

        # Step 2: Upsample W dimension (W -> 2*W)
        # x: [1, C, H2, W] -> [C*H2, W]
        x = ttnn.reshape(x, [C * H2, W], memory_config=DRAM)
        # matmul with mask [W, W2] transposed -> output [C*H2, W2]
        x = ttnn.matmul(
            x,
            self.upsample_mask_w,
            memory_config=DRAM,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        # Reshape to [1, C, H2, W2]
        x = ttnn.reshape(x, [1, C, H2, W2], memory_config=DRAM)


        # Apply upsample conv (3x3, same channels)
        x = conv2d(
            x,
            self.conv_weight,
            self.conv_bias,
            self.device,
            C,
            C,
            H2,
            W2,
            act_block_h_override=self.act_block_h_override,
        )
        return x


# ---------------------------------------------------------------------------
# UpBlock
# ---------------------------------------------------------------------------


class UpBlock(LightweightModule):
    """Up-sampling decoder block: N resnets + optional Upsample.

    Weight name prefix example: "decoder.up_blocks.0"
    """

    def __init__(self, resnets, upsample=None):
        super().__init__()
        self.resnets = resnets
        self.upsample = upsample

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet.forward(x)
        if self.upsample is not None:
            x = self.upsample.forward(x)
        return x


# ---------------------------------------------------------------------------
# MidBlock
# ---------------------------------------------------------------------------


class MidBlock(LightweightModule):
    """Mid-block: resnet0 -> attention -> resnet1.

    Weight name prefix: "decoder.mid_block"
    """

    def __init__(self, resnet0, attention, resnet1):
        super().__init__()
        self.resnet0 = resnet0
        self.attention = attention
        self.resnet1 = resnet1

    def forward(self, x):
        x = self.resnet0.forward(x)
        x = self.attention.forward(x)
        x = self.resnet1.forward(x)
        return x


# ---------------------------------------------------------------------------
# VaeDecoderTTNN
# ---------------------------------------------------------------------------


class VaeDecoderTTNN(LightweightModule):
    """Full VAE decoder in TTNN.

    Spatial dimensions through the network:
        Input:       [1, 16, 160, 90]
        conv_in:     [1, 512, 160, 90]
        mid_block:   [1, 512, 160, 90]
        up_block[0]: [1, 512, 160, 90] -> upsample -> [1, 512, 320, 180]
        up_block[1]: [1, 512, 320, 180] -> upsample -> [1, 512, 640, 360]
        up_block[2]: [1, 256, 640, 360] -> upsample -> [1, 256, 1280, 720]
        up_block[3]: [1, 128, 1280, 720]
        conv_out:    [1, 3, 1280, 720]
    """

    def __init__(
        self,
        conv_in_weight,
        conv_in_bias,
        mid_block,
        up_blocks,
        conv_norm_out_weight_ce,
        conv_norm_out_bias_ce,
        conv_out_weight,
        conv_out_bias,
        device,
    ):
        super().__init__()
        self.conv_in_weight = conv_in_weight
        self.conv_in_bias = conv_in_bias
        self.mid_block = mid_block
        self.up_blocks = up_blocks
        self.conv_norm_out_weight_ce = conv_norm_out_weight_ce
        self.conv_norm_out_bias_ce = conv_norm_out_bias_ce
        self.conv_out_weight = conv_out_weight
        self.conv_out_bias = conv_out_bias
        self.device = device

    def forward(self, x):
        """Forward pass.

        Args:
            x: [1, 16, 160, 90] BFLOAT16 on device.

        Returns:
            [1, 3, 1280, 720] BFLOAT16 on device.
        """
        # conv_in: [1, 16, 160, 90] -> [1, 512, 160, 90]
        x = conv2d(
            x,
            self.conv_in_weight,
            self.conv_in_bias,
            self.device,
            in_channels=16,
            out_channels=512,
            height=160,
            width=90,
            act_block_h_override=0,
        )


        # mid_block: [1, 512, 160, 90]
        x = self.mid_block.forward(x)


        # up_blocks
        for up_block in self.up_blocks:
            x = up_block.forward(x)

        # conv_norm_out + SiLU + conv_out
        # At this point x is [1, 128, 1280, 720]
        x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM)
        x = group_norm(
            x,
            self.conv_norm_out_weight_ce,
            self.conv_norm_out_bias_ce,
            128,
            1280,
            720,
            self.device,
        )
        x = ttnn.silu(x, memory_config=DRAM)
        x = ttnn.typecast(x, ttnn.DataType.BFLOAT16, memory_config=DRAM)

        # conv_out: [1, 128, 1280, 720] -> [1, 3, 1280, 720]
        x = conv2d(
            x,
            self.conv_out_weight,
            self.conv_out_bias,
            self.device,
            in_channels=128,
            out_channels=3,
            height=1280,
            width=720,
            act_block_h_override=1024,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
            ),
        )
        return x


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_weights_from_pytorch(state_dict, device):
    """Load PyTorch state dict into TTNN tensors following tensor_load_config.json.

    Args:
        state_dict: PyTorch state dict with keys like "conv_in.weight" (no "decoder." prefix).
        device: TTNN device.

    Returns:
        dict of {name: ttnn.Tensor} with "decoder." prefix in keys.
    """
    config_path = os.path.join(os.path.dirname(__file__), "tensor_load_config.json")
    with open(config_path) as f:
        load_config = json.load(f)

    weights = {}
    for key, cfg in load_config.items():
        if key == "args_0":
            continue  # Not a weight

        # Strip "decoder." prefix to look up in PyTorch state dict
        pt_key = key.replace("decoder.", "", 1)
        if pt_key not in state_dict:
            print(f"[load_weights] Warning: {pt_key} not in state_dict, skipping")
            continue

        pt_tensor = state_dict[pt_key]

        # Convert to TTNN tensor
        layout = ttnn.Layout.TILE if cfg["layout"] == "TILE" else ttnn.Layout.ROW_MAJOR
        t = ttnn.from_torch(pt_tensor, dtype=ttnn.DataType.BFLOAT16, layout=layout)

        if cfg["on_device"]:
            t = ttnn.to_device(t, device, memory_config=DRAM)

        weights[key] = t

    return weights


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(weights, device):
    """Build the VaeDecoderTTNN from loaded weights.

    This function:
    1. Runs const-evals on norm weights (prepare_norm_weight, prepare_attn_norm_weight)
    2. Prepares conv weights and biases (prepare_conv_weight, prepare_conv_bias)
    3. Creates upsample masks
    4. Assembles all LightweightModule instances

    Args:
        weights: dict from load_weights_from_pytorch
        device: TTNN device

    Returns:
        VaeDecoderTTNN instance ready for forward()
    """
    W = weights  # alias for brevity

    # -----------------------------------------------------------------------
    # Helper: prepare a resnet block's weights
    # -----------------------------------------------------------------------
    def _prep_resnet(prefix, in_ch, out_ch, H, W_dim, act_block_h=1024,
                     has_shortcut=False, shortcut_act_block_h=0):
        """Prepare all weights for a ResNetBlock."""
        n1w_ce = prepare_norm_weight(W[f"{prefix}.norm1.weight"], in_ch, NUM_GROUPS, device)
        n1b_ce = prepare_norm_weight(W[f"{prefix}.norm1.bias"], in_ch, NUM_GROUPS, device)
        n2w_ce = prepare_norm_weight(W[f"{prefix}.norm2.weight"], out_ch, NUM_GROUPS, device)
        n2b_ce = prepare_norm_weight(W[f"{prefix}.norm2.bias"], out_ch, NUM_GROUPS, device)

        c1w = prepare_conv_weight(
            W[f"{prefix}.conv1.weight"], device,
            in_channels=in_ch, out_channels=out_ch, batch_size=1,
            input_height=H, input_width=W_dim,
            kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
            dilation=[1, 1], groups=1, has_bias=True,
            act_block_h_override=act_block_h,
        )
        c1b = prepare_conv_bias(
            W[f"{prefix}.conv1.bias"], device,
            in_channels=in_ch, out_channels=out_ch, batch_size=1,
            input_height=H, input_width=W_dim,
            kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
            dilation=[1, 1], groups=1,
            act_block_h_override=act_block_h,
        )
        c2w = prepare_conv_weight(
            W[f"{prefix}.conv2.weight"], device,
            in_channels=out_ch, out_channels=out_ch, batch_size=1,
            input_height=H, input_width=W_dim,
            kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
            dilation=[1, 1], groups=1, has_bias=True,
            act_block_h_override=act_block_h,
        )
        c2b = prepare_conv_bias(
            W[f"{prefix}.conv2.bias"], device,
            in_channels=out_ch, out_channels=out_ch, batch_size=1,
            input_height=H, input_width=W_dim,
            kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
            dilation=[1, 1], groups=1,
            act_block_h_override=act_block_h,
        )

        sc_w, sc_b = None, None
        if has_shortcut:
            sc_w = prepare_conv_weight(
                W[f"{prefix}.conv_shortcut.weight"], device,
                in_channels=in_ch, out_channels=out_ch, batch_size=1,
                input_height=H, input_width=W_dim,
                kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0],
                dilation=[1, 1], groups=1, has_bias=True,
                act_block_h_override=shortcut_act_block_h,
            )
            sc_b = prepare_conv_bias(
                W[f"{prefix}.conv_shortcut.bias"], device,
                in_channels=in_ch, out_channels=out_ch, batch_size=1,
                input_height=H, input_width=W_dim,
                kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0],
                dilation=[1, 1], groups=1,
                act_block_h_override=shortcut_act_block_h,
            )

        return ResNetBlock(
            norm1_weight_ce=n1w_ce,
            norm1_bias_ce=n1b_ce,
            conv1_weight=c1w,
            conv1_bias=c1b,
            norm2_weight_ce=n2w_ce,
            norm2_bias_ce=n2b_ce,
            conv2_weight=c2w,
            conv2_bias=c2b,
            in_channels=in_ch,
            out_channels=out_ch,
            height=H,
            width=W_dim,
            act_block_h_override=act_block_h,
            conv_shortcut_weight=sc_w,
            conv_shortcut_bias=sc_b,
            conv_shortcut_act_block_h=shortcut_act_block_h,
            device=device,
        )

    # -----------------------------------------------------------------------
    # Helper: prepare an upsample block's weights
    # -----------------------------------------------------------------------
    def _prep_upsample(prefix, channels, in_h, in_w, act_block_h=1024):
        out_h = in_h * 2
        out_w = in_w * 2
        mask_h = create_upsample_mask(in_h, out_h, device)
        mask_w = create_upsample_mask(in_w, out_w, device)
        conv_w = prepare_conv_weight(
            W[f"{prefix}.conv.weight"], device,
            in_channels=channels, out_channels=channels, batch_size=1,
            input_height=out_h, input_width=out_w,
            kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
            dilation=[1, 1], groups=1, has_bias=True,
            act_block_h_override=act_block_h,
        )
        conv_b = prepare_conv_bias(
            W[f"{prefix}.conv.bias"], device,
            in_channels=channels, out_channels=channels, batch_size=1,
            input_height=out_h, input_width=out_w,
            kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
            dilation=[1, 1], groups=1,
            act_block_h_override=act_block_h,
        )
        return Upsample(
            conv_weight=conv_w,
            conv_bias=conv_b,
            channels=channels,
            in_height=in_h,
            in_width=in_w,
            upsample_mask_h=mask_h,
            upsample_mask_w=mask_w,
            act_block_h_override=act_block_h,
            device=device,
        )

    # -----------------------------------------------------------------------
    # conv_in
    # -----------------------------------------------------------------------
    print("[build_model] Preparing conv_in...")
    conv_in_weight = prepare_conv_weight(
        W["decoder.conv_in.weight"], device,
        in_channels=16, out_channels=512, batch_size=1,
        input_height=160, input_width=90,
        kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
        dilation=[1, 1], groups=1, has_bias=True,
        act_block_h_override=0,
    )
    conv_in_bias = prepare_conv_bias(
        W["decoder.conv_in.bias"], device,
        in_channels=16, out_channels=512, batch_size=1,
        input_height=160, input_width=90,
        kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
        dilation=[1, 1], groups=1,
        act_block_h_override=0,
    )

    # -----------------------------------------------------------------------
    # mid_block: resnets.0, attention, resnets.1  (all at 160x90, 512 channels)
    # -----------------------------------------------------------------------
    print("[build_model] Preparing mid_block...")
    mid_resnet0 = _prep_resnet(
        "decoder.mid_block.resnets.0", 512, 512, 160, 90, act_block_h=1024
    )
    mid_resnet1 = _prep_resnet(
        "decoder.mid_block.resnets.1", 512, 512, 160, 90, act_block_h=1024
    )

    # Attention block
    seq_len = 160 * 90  # 14400
    attn_gn_w_ce = prepare_attn_norm_weight(
        W["decoder.mid_block.attentions.0.group_norm.weight"], 512, NUM_GROUPS, device
    )
    attn_gn_b_ce = prepare_attn_norm_weight(
        W["decoder.mid_block.attentions.0.group_norm.bias"], 512, NUM_GROUPS, device
    )
    q_bias_ce = prepare_attn_bias(
        W["decoder.mid_block.attentions.0.to_q.bias"], 512, seq_len, device
    )
    k_bias_ce = prepare_attn_bias(
        W["decoder.mid_block.attentions.0.to_k.bias"], 512, seq_len, device
    )
    v_bias_ce = prepare_attn_bias(
        W["decoder.mid_block.attentions.0.to_v.bias"], 512, seq_len, device
    )
    out_bias_ce = prepare_attn_bias(
        W["decoder.mid_block.attentions.0.to_out.0.bias"], 512, seq_len, device
    )

    attention = AttentionBlock(
        group_norm_weight_ce=attn_gn_w_ce,
        group_norm_bias_ce=attn_gn_b_ce,
        q_weight=W["decoder.mid_block.attentions.0.to_q.weight"],
        q_bias_ce=q_bias_ce,
        k_weight=W["decoder.mid_block.attentions.0.to_k.weight"],
        k_bias_ce=k_bias_ce,
        v_weight=W["decoder.mid_block.attentions.0.to_v.weight"],
        v_bias_ce=v_bias_ce,
        out_weight=W["decoder.mid_block.attentions.0.to_out.0.weight"],
        out_bias_ce=out_bias_ce,
        channels=512,
        height=160,
        width=90,
        device=device,
    )

    mid_block = MidBlock(mid_resnet0, attention, mid_resnet1)

    # -----------------------------------------------------------------------
    # up_blocks[0]: 512->512, 160x90, upsample to 320x180
    # -----------------------------------------------------------------------
    print("[build_model] Preparing up_block[0]...")
    ub0_resnets = [
        _prep_resnet("decoder.up_blocks.0.resnets.0", 512, 512, 160, 90, act_block_h=1024),
        _prep_resnet("decoder.up_blocks.0.resnets.1", 512, 512, 160, 90, act_block_h=1024),
        _prep_resnet("decoder.up_blocks.0.resnets.2", 512, 512, 160, 90, act_block_h=1024),
    ]
    ub0_upsample = _prep_upsample(
        "decoder.up_blocks.0.upsamplers.0", 512, 160, 90, act_block_h=1024
    )
    up_block_0 = UpBlock(ub0_resnets, ub0_upsample)

    # -----------------------------------------------------------------------
    # up_blocks[1]: 512->512, 320x180, upsample to 640x360
    # -----------------------------------------------------------------------
    print("[build_model] Preparing up_block[1]...")
    ub1_resnets = [
        _prep_resnet("decoder.up_blocks.1.resnets.0", 512, 512, 320, 180, act_block_h=1024),
        _prep_resnet("decoder.up_blocks.1.resnets.1", 512, 512, 320, 180, act_block_h=1024),
        _prep_resnet("decoder.up_blocks.1.resnets.2", 512, 512, 320, 180, act_block_h=1024),
    ]
    ub1_upsample = _prep_upsample(
        "decoder.up_blocks.1.upsamplers.0", 512, 320, 180, act_block_h=1024
    )
    up_block_1 = UpBlock(ub1_resnets, ub1_upsample)

    # -----------------------------------------------------------------------
    # up_blocks[2]: 512->256, 640x360, upsample to 1280x720
    # First resnet has conv_shortcut (512->256)
    # -----------------------------------------------------------------------
    print("[build_model] Preparing up_block[2]...")
    ub2_resnets = [
        _prep_resnet(
            "decoder.up_blocks.2.resnets.0", 512, 256, 640, 360,
            act_block_h=1024, has_shortcut=True, shortcut_act_block_h=0
        ),
        _prep_resnet("decoder.up_blocks.2.resnets.1", 256, 256, 640, 360, act_block_h=1024),
        _prep_resnet("decoder.up_blocks.2.resnets.2", 256, 256, 640, 360, act_block_h=1024),
    ]
    ub2_upsample = _prep_upsample(
        "decoder.up_blocks.2.upsamplers.0", 256, 640, 360, act_block_h=1024
    )
    up_block_2 = UpBlock(ub2_resnets, ub2_upsample)

    # -----------------------------------------------------------------------
    # up_blocks[3]: 256->128, 1280x720, no upsample
    # First resnet has conv_shortcut (256->128)
    # -----------------------------------------------------------------------
    print("[build_model] Preparing up_block[3]...")
    ub3_resnets = [
        _prep_resnet(
            "decoder.up_blocks.3.resnets.0", 256, 128, 1280, 720,
            act_block_h=1024, has_shortcut=True, shortcut_act_block_h=0
        ),
        _prep_resnet("decoder.up_blocks.3.resnets.1", 128, 128, 1280, 720, act_block_h=1024),
        _prep_resnet("decoder.up_blocks.3.resnets.2", 128, 128, 1280, 720, act_block_h=1024),
    ]
    up_block_3 = UpBlock(ub3_resnets, upsample=None)

    # -----------------------------------------------------------------------
    # conv_norm_out + conv_out
    # -----------------------------------------------------------------------
    print("[build_model] Preparing conv_norm_out and conv_out...")
    conv_norm_out_w_ce = prepare_norm_weight(
        W["decoder.conv_norm_out.weight"], 128, NUM_GROUPS, device
    )
    conv_norm_out_b_ce = prepare_norm_weight(
        W["decoder.conv_norm_out.bias"], 128, NUM_GROUPS, device
    )
    conv_out_weight = prepare_conv_weight(
        W["decoder.conv_out.weight"], device,
        in_channels=128, out_channels=3, batch_size=1,
        input_height=1280, input_width=720,
        kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
        dilation=[1, 1], groups=1, has_bias=True,
        act_block_h_override=1024,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
    )
    conv_out_bias = prepare_conv_bias(
        W["decoder.conv_out.bias"], device,
        in_channels=128, out_channels=3, batch_size=1,
        input_height=1280, input_width=720,
        kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1],
        dilation=[1, 1], groups=1,
        act_block_h_override=1024,
    )

    # -----------------------------------------------------------------------
    # Assemble full model
    # -----------------------------------------------------------------------
    print("[build_model] Assembling VaeDecoderTTNN...")
    model = VaeDecoderTTNN(
        conv_in_weight=conv_in_weight,
        conv_in_bias=conv_in_bias,
        mid_block=mid_block,
        up_blocks=[up_block_0, up_block_1, up_block_2, up_block_3],
        conv_norm_out_weight_ce=conv_norm_out_w_ce,
        conv_norm_out_bias_ce=conv_norm_out_b_ce,
        conv_out_weight=conv_out_weight,
        conv_out_bias=conv_out_bias,
        device=device,
    )

    print("[build_model] Done.")
    return model
