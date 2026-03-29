# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained DynUNet implementation compatible with MONAI's state dict
# key structure. This avoids a runtime dependency on the monai package which
# requires Python >= 3.9.
#
# Architecture reference:
#   MONAI DynUNet (Apache 2.0) - https://github.com/Project-MONAI/MONAI
"""
Standalone DynUNet for vesselFM 3D blood vessel segmentation.
"""
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def _get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def _get_output_padding(kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def _conv_layer(
    spatial_dims,
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    bias=False,
    is_transposed=False,
):
    """Create a conv layer wrapped in nn.Sequential with child named 'conv',
    matching MONAI's Convolution state dict key structure."""
    padding = _get_padding(kernel_size, stride)
    seq = nn.Sequential()
    if is_transposed:
        output_padding = _get_output_padding(kernel_size, stride, padding)
        conv = (
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            )
            if spatial_dims == 3
            else nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            )
        )
    else:
        conv = (
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
            if spatial_dims == 3
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )
    seq.add_module("conv", conv)
    return seq


def _norm_layer(spatial_dims, channels):
    """Instance norm with affine=True, matching MONAI default."""
    if spatial_dims == 3:
        return nn.InstanceNorm3d(channels, affine=True)
    return nn.InstanceNorm2d(channels, affine=True)


class UnetResBlock(nn.Module):
    def __init__(
        self, spatial_dims, in_channels, out_channels, kernel_size, stride, **kwargs
    ):
        super().__init__()
        self.conv1 = _conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = _conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = _norm_layer(spatial_dims, out_channels)
        self.norm2 = _norm_layer(spatial_dims, out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = _conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride
            )
            self.norm3 = _norm_layer(spatial_dims, out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(nn.Module):
    def __init__(
        self, spatial_dims, in_channels, out_channels, kernel_size, stride, **kwargs
    ):
        super().__init__()
        self.conv1 = _conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = _conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = _norm_layer(spatial_dims, out_channels)
        self.norm2 = _norm_layer(spatial_dims, out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        upsample_kernel_size,
        trans_bias=False,
        **kwargs
    ):
        super().__init__()
        self.transp_conv = _conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            bias=trans_bias,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels):
        super().__init__()
        self.conv = _conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, bias=True
        )

    def forward(self, inp):
        return self.conv(inp)


class DynUNetSkipLayer(nn.Module):
    heads: Optional[List[torch.Tensor]]

    def __init__(
        self, index, downsample, upsample, next_layer, heads=None, super_head=None
    ):
        super().__init__()
        self.downsample = downsample
        self.next_layer = next_layer
        self.upsample = upsample
        self.super_head = super_head
        self.heads = heads
        self.index = index

    def forward(self, x):
        downout = self.downsample(x)
        nextout = self.next_layer(downout)
        upout = self.upsample(nextout, downout)
        if self.super_head is not None and self.heads is not None and self.index > 0:
            self.heads[self.index - 1] = self.super_head(upout)
        return upout


class DynUNet(nn.Module):
    """Standalone DynUNet matching MONAI's architecture and state dict key structure."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        res_block: bool = False,
        trans_bias: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.trans_bias = trans_bias
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        if filters is not None:
            self.filters = list(filters[: len(strides)])
        else:
            self.filters = [
                min(2 ** (5 + i), 320 if spatial_dims == 3 else 512)
                for i in range(len(strides))
            ]

        self.input_block = self._get_input_block()
        self.downsamples = self._get_downsamples()
        self.bottleneck = self._get_bottleneck()
        self.upsamples = self._get_upsamples()
        self.output_block = UnetOutBlock(spatial_dims, self.filters[0], out_channels)

        self.apply(self._initialize_weights)

        def create_skips(index, downsamples, upsamples, bottleneck):
            if len(downsamples) == 0:
                return bottleneck
            next_layer = create_skips(
                1 + index, downsamples[1:], upsamples[1:], bottleneck
            )
            return DynUNetSkipLayer(
                index,
                downsample=downsamples[0],
                upsample=upsamples[0],
                next_layer=next_layer,
            )

        self.skip_layers = create_skips(
            0,
            [self.input_block] + list(self.downsamples),
            self.upsamples[::-1],
            self.bottleneck,
        )

    def forward(self, x):
        out = self.skip_layers(x)
        out = self.output_block(out)
        return out

    def _get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
        )

    def _get_bottleneck(self):
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
        )

    def _get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        layers = []
        for in_c, out_c, kernel, stride in zip(inp, out, kernel_size, strides):
            layers.append(
                self.conv_block(self.spatial_dims, in_c, out_c, kernel, stride)
            )
        return nn.ModuleList(layers)

    def _get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        layers = []
        for in_c, out_c, kernel, stride, up_kernel in zip(
            inp, out, kernel_size, strides, upsample_kernel_size
        ):
            layers.append(
                UnetUpBlock(
                    self.spatial_dims,
                    in_c,
                    out_c,
                    kernel_size=kernel,
                    stride=stride,
                    upsample_kernel_size=up_kernel,
                    trans_bias=self.trans_bias,
                )
            )
        return nn.ModuleList(layers)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(
            module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
