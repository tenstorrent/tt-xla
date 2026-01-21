# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
# code apapted from :
# https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2

MIT License

Copyright (c) 2021 Yonghye Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

"""
CPU implementation of DCNv2 (Modulated Deformable Convolution) to remove the
dependency on the DCNv2 used in the CenterNet implementation for ResNet and DLA-based variants:
https://github.com/xingyizhou/CenterNet/tree/master/src/lib/models/networks/DCNv2

This implementation avoids the need to build and compile custom CUDA/C++ extensions.

"""

import torch
import torch.nn as nn
import torchvision


class DCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCN, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.deformable_groups = deformable_groups

        # Generate offset + mask (for modulated DCN)
        offset_mask_channels = 3 * deformable_groups * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            offset_mask_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.init_offset()

        # Deformable convolution weight and bias
        self.weight = nn.Parameter(
            torch.Tensor(
                out_channels,
                in_channels // deformable_groups,
                *kernel_size,
            )
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)

    def init_offset(self):
        nn.init.constant_(self.conv_offset_mask.weight, 0)
        nn.init.constant_(self.conv_offset_mask.bias, 0)

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)

        # Split into offset and mask
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)  # Ensure mask values in [0, 1]
        input_dtype = x.dtype
        if input_dtype is torch.bfloat16:
            x_fp32 = x.to(dtype=torch.float32)
            offset_fp32 = offset.to(dtype=torch.float32)
            weight_fp32 = self.weight.to(dtype=torch.float32)
            bias_fp32 = (
                self.bias.to(dtype=torch.float32) if self.bias is not None else None
            )
            out = torchvision.ops.deform_conv2d(
                input=x_fp32,
                offset=offset_fp32,
                weight=weight_fp32,
                bias=bias_fp32,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask.to(dtype=torch.float32),
            ).to(dtype=input_dtype)
        else:
            out = torchvision.ops.deform_conv2d(
                input=x,
                offset=offset,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask,
            )
        return out
