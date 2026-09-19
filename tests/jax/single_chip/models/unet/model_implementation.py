# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Union

import flax.linen as nn
import jax.numpy as jnp
from jax._src.typing import DTypeLike


class DoubleConv(nn.Module):
    out_channels: int
    activation: Callable = nn.relu
    use_batchnorm: bool = False
    param_dtype: Optional[DTypeLike] = jnp.bfloat16

    def setup(self):
        self.conv1 = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            padding="VALID",
            use_bias=not self.use_batchnorm,
            param_dtype=self.param_dtype,
        )
        self.conv2 = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            padding="VALID",
            use_bias=not self.use_batchnorm,
            param_dtype=self.param_dtype,
        )
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

    def __call__(self, x, train: bool):
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x, use_running_average=not train)
        x = self.activation(x)

        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x, use_running_average=not train)
        x = self.activation(x)

        return x


class DownBlock(nn.Module):
    out_channels: int
    activation: Callable = nn.relu
    use_batchnorm: bool = False
    param_dtype: Optional[DTypeLike] = jnp.bfloat16

    def setup(self):
        self.double_conv = DoubleConv(
            out_channels=self.out_channels,
            activation=self.activation,
            use_batchnorm=self.use_batchnorm,
            param_dtype=self.param_dtype,
        )

    def __call__(self, x, train: bool):
        feature_map = self.double_conv(x, train)
        pooled_x = nn.max_pool(
            feature_map, window_shape=(2, 2), strides=(2, 2), padding="VALID"
        )
        return pooled_x, feature_map


class UpBlock(nn.Module):
    out_channels: int
    activation: Callable = nn.relu
    use_batchnorm: bool = False
    param_dtype: Optional[DTypeLike] = jnp.bfloat16

    def setup(self):
        self.up_conv = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="VALID",
            param_dtype=self.param_dtype,
        )
        self.double_conv = DoubleConv(
            out_channels=self.out_channels,
            activation=self.activation,
            use_batchnorm=self.use_batchnorm,
            param_dtype=self.param_dtype,
        )

    def __call__(self, x, skip, train: bool):
        x = self.up_conv(x)
        skip = self.crop_skip_connection(skip, x.shape)
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.double_conv(x, train)
        return x

    @staticmethod
    def crop_skip_connection(skip: jnp.ndarray, target_shape: tuple) -> jnp.ndarray:
        """
        After upsampling with ConvTranspose, 'x' will not match the size of the
        corresponding skip connection because the convolutions are applied unpadded,
        which results in lower image size after each convolution. To concatenate them
        correctly, we center-crop 'skip' to match 'x' spatially. Then, we concatenate
        'x' and the cropped 'skip' along the channel axis before applying DoubleConv.
        """
        diff_h = skip.shape[1] - target_shape[1]
        diff_w = skip.shape[2] - target_shape[2]
        start_h = diff_h // 2
        start_w = diff_w // 2
        return skip[
            :,
            start_h : start_h + target_shape[1],
            start_w : start_w + target_shape[2],
            :,
        ]


class UNet(nn.Module):
    in_channels: int = 1
    out_channels: int = 2
    hidden_channels: int = 64
    num_levels: int = 4
    activation: Callable = nn.relu
    use_batchnorm: bool = False
    param_dtype: Optional[DTypeLike] = jnp.bfloat16

    def setup(self):
        down_blocks = []
        out_ch = self.in_channels * self.hidden_channels

        # Create down blocks
        for i in range(self.num_levels):
            down_blocks.append(
                DownBlock(
                    out_channels=out_ch,
                    activation=self.activation,
                    use_batchnorm=self.use_batchnorm,
                    param_dtype=self.param_dtype,
                )
            )
            out_ch = out_ch * 2

        # Bottom convolution
        self.bottom_double_conv = DoubleConv(
            out_channels=out_ch,
            activation=self.activation,
            use_batchnorm=self.use_batchnorm,
            param_dtype=self.param_dtype,
        )

        # Create up blocks
        up_blocks = []
        for i in range(self.num_levels):
            out_ch = out_ch // 2
            up_blocks.append(
                UpBlock(
                    out_channels=out_ch,
                    activation=self.activation,
                    use_batchnorm=self.use_batchnorm,
                    param_dtype=self.param_dtype,
                )
            )

        self.down_blocks = tuple(down_blocks)
        self.up_blocks = tuple(up_blocks)

        # Final output convolution
        self.output_conv = nn.Conv(
            self.out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            param_dtype=self.param_dtype,
        )

    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        skip_connections = []

        for down_block in self.down_blocks:
            x, skip = down_block(x, train)
            skip_connections.append(skip)

        x = self.bottom_double_conv(x, train)

        for up_block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up_block(x, skip, train)

        return self.output_conv(x)
