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

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="VALID")(x)
        x = self.activation(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="VALID")(x)
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
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            activation=self.activation,
        )

        down_blocks = []
        in_ch = self.hidden_channels
        for i in range(self.num_levels - 1):
            out_ch = self.hidden_channels * 2 ** (i + 1)
            block = DoubleConv(
                in_channels=in_ch, out_channels=out_ch, activation=self.activation
            )
            down_blocks.append(block)
            in_ch = out_ch
        self.down_sampling_blocks = tuple(down_blocks)

        self.bottleneck = DoubleConv(
            in_channels=in_ch, out_channels=in_ch * 2, activation=self.activation
        )

        up_blocks = []
        in_ch = in_ch * 2
        for i in reversed(range(self.num_levels)):
            out_ch = self.hidden_channels * 2**i
            upconv = nn.ConvTranspose(
                features=out_ch, kernel_size=(2, 2), strides=(2, 2)
            )
            double_conv = DoubleConv(
                in_channels=in_ch, out_channels=out_ch, activation=self.activation
            )
            up_blocks.append((upconv, double_conv))
            in_ch = out_ch
        self.up_sampling_blocks = tuple(up_blocks)

        self.output_conv = nn.Conv(
            features=self.out_channels, kernel_size=(1, 1), strides=(1, 1)
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = self.lifting(x)

        skip_connections = [x]
        for i, down in enumerate(self.down_sampling_blocks):
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = down(x)
            skip_connections.append(x)

        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self.bottleneck(x)

        for i, (up, conv) in enumerate(self.up_sampling_blocks):
            x = up(x)
            skip = skip_connections[-(i + 1)]
            diff_h = skip.shape[1] - x.shape[1]
            diff_w = skip.shape[2] - x.shape[2]
            start_h = diff_h // 2
            start_w = diff_w // 2
            skip = skip[
                :, start_h : start_h + x.shape[1], start_w : start_w + x.shape[2], :
            ]
            x = jnp.concatenate([x, skip], axis=-1)
            x = conv(x)

        x = self.output_conv(x)

        return x
