# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Tuple
import flax.linen as nn
import jax.numpy as jnp


class DoubleConv(nn.Module):
    in_channels: int
    out_channels: int
    activation: Callable = nn.relu

    def setup(self):
        self.conv1 = nn.Conv(
            self.out_channels, kernel_size=(3, 3), padding="VALID", use_bias=False
        )
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv(
            self.out_channels, kernel_size=(3, 3), padding="VALID", use_bias=False
        )
        self.bn2 = nn.BatchNorm()

    def __call__(self, x, train: bool):
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = self.activation(x)

        return x


class UNet(nn.Module):
    in_channels: int
    out_channels: int
    hidden_channels: int
    num_levels: int
    activation: Callable = nn.relu

    def setup(self):
        self.lifting = DoubleConv(
            self.in_channels, self.hidden_channels, self.activation
        )

        down = []
        in_ch = self.hidden_channels
        for i in range(self.num_levels - 1):
            out_ch = self.hidden_channels * 2 ** (i + 1)
            down.append(DoubleConv(in_ch, out_ch, self.activation))
            in_ch = out_ch

        down.append(DoubleConv(in_ch, in_ch * 2, self.activation))
        in_ch = in_ch * 2
        self.down = tuple(down)

        up_blocks = []
        up_convs = []
        for i in reversed(range(self.num_levels)):
            out_ch = self.hidden_channels * 2**i
            up_convs.append(
                nn.ConvTranspose(
                    features=out_ch, kernel_size=(2, 2), strides=(2, 2), padding="VALID"
                )
            )
            up_blocks.append(DoubleConv(in_ch, out_ch, self.activation))
            in_ch = out_ch

        self.up_blocks = tuple(up_blocks)
        self.up_convs = tuple(up_convs)

        self.output_conv = nn.Conv(
            self.out_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID"
        )

    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        skips = []
        x = self.lifting(x, train)
        skips.append(x)

        for block in self.down[:-1]:
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
            x = block(x, train)
            skips.append(x)

        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = self.down[-1](x, train)

        """
        After upsampling with ConvTranspose, 'x' might not perfectly match the size
        of the corresponding skip connection. To concatenate them correctly, we
        center-crop 'skip' to match 'x' spatially. Then, we concatenate 'x' and the
        cropped 'skip' along the channel axis before applying DoubleConv.
        """
        for i in range(self.num_levels):
            x = self.up_convs[i](x)
            skip = skips[-(i + 1)]
            diff_h = skip.shape[1] - x.shape[1]
            diff_w = skip.shape[2] - x.shape[2]
            start_h = diff_h // 2
            start_w = diff_w // 2
            skip = skip[
                :, start_h : start_h + x.shape[1], start_w : start_w + x.shape[2], :
            ]
            x = jnp.concatenate([x, skip], axis=-1)
            x = self.up_blocks[i](x, train)

        x = self.output_conv(x)

        return x
