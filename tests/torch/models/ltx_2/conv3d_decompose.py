# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Conv3d -> Conv2d temporal decomposition for tt-metal compatibility.

The tt-metal Conv3dDeviceOperation allocates circular buffers that exceed
per-core L1 capacity (1.57 MB) for typical LTX-2 channel dimensions.
This module decomposes Conv3d into a sum of Conv2d operations over the
temporal kernel dimension, avoiding the Conv3d kernel entirely.

Math:
  Conv3d(x, w, b)[..., t_out, :, :] =
    sum_{k=0}^{k_t-1} Conv2d(x[..., t_in+k, :, :], w[:,:,k,:,:]) + b

Usage:
  from conv3d_decompose import patch_conv3d_to_conv2d
  patch_conv3d_to_conv2d()  # Call before model creation or .to(device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3d_as_conv2d(input, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0),
                     dilation=(1, 1, 1), groups=1):
    """Drop-in replacement for F.conv3d using temporal-slice Conv2d decomposition.

    Avoids tt-metal Conv3dDeviceOperation L1 overflow by decomposing the 3D
    convolution into a sum of 2D convolutions over the temporal kernel dimension.
    """
    # Normalize to tuples
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    B, C_in, T_in, H_in, W_in = input.shape
    C_out, C_in_g, k_t, k_h, k_w = weight.shape
    stride_t, stride_h, stride_w = stride
    pad_t, pad_h, pad_w = padding
    dil_t, dil_h, dil_w = dilation

    # Effective kernel size with dilation
    eff_k_t = (k_t - 1) * dil_t + 1

    # Output temporal size
    T_out = (T_in + 2 * pad_t - eff_k_t) // stride_t + 1

    # Collect output frames
    frames = []
    for t_out in range(T_out):
        t_in_start = t_out * stride_t - pad_t
        acc = None
        for k in range(k_t):
            t_in = t_in_start + k * dil_t
            if 0 <= t_in < T_in:
                frame = input[:, :, t_in, :, :]  # [B, C_in, H, W]
                w_2d = weight[:, :, k, :, :]      # [C_out, C_in_g, k_h, k_w]
                b_2d = bias if (k == 0 and acc is None) else None
                out = F.conv2d(frame, w_2d, bias=b_2d, stride=(stride_h, stride_w),
                               padding=(pad_h, pad_w), dilation=(dil_h, dil_w), groups=groups)
                acc = out if acc is None else acc + out
            elif k == 0 and acc is None and bias is not None:
                # First kernel position is out of bounds but we need to initialize bias
                # Will be handled when we find an in-bounds position
                pass

        if acc is None:
            # All kernel positions were out of bounds (shouldn't happen with valid padding)
            # Initialize with zeros + bias
            H_out = (H_in + 2 * pad_h - ((k_h - 1) * dil_h + 1)) // stride_h + 1
            W_out = (W_in + 2 * pad_w - ((k_w - 1) * dil_w + 1)) // stride_w + 1
            acc = torch.zeros(B, C_out, H_out, W_out, dtype=input.dtype, device=input.device)
            if bias is not None:
                acc = acc + bias.reshape(1, -1, 1, 1)
        elif bias is not None and True:
            # Check if bias was already added (it's added with the first in-bounds k)
            # We need to check if the first k=0 was in-bounds
            t_first = t_in_start
            if t_first < 0 or t_first >= T_in:
                # Bias wasn't added yet because first position was out of bounds
                acc = acc + bias.reshape(1, -1, 1, 1)

        frames.append(acc)

    return torch.stack(frames, dim=2)  # [B, C_out, T_out, H_out, W_out]


def patch_conv3d_to_conv2d():
    """Monkey-patch nn.Conv3d.forward to use Conv2d temporal decomposition.

    This avoids the tt-metal Conv3dDeviceOperation L1 overflow by converting
    all Conv3d operations into equivalent sequences of Conv2d operations.

    Call this BEFORE creating models that use Conv3d.
    """
    _original_forward = nn.Conv3d.forward

    def _decomposed_forward(self, input):
        return conv3d_as_conv2d(
            input, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )

    nn.Conv3d.forward = _decomposed_forward
    return _original_forward  # Return original for cleanup if needed
