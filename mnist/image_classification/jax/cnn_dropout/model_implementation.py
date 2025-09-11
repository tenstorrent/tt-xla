# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from flax import linen as nn


class MNISTCNNDropoutModel(nn.Module):
    """MNIST CNN model implementation with dropout."""

    @nn.compact
    def __call__(self, x, *, train: bool):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Dropout(rate=0.25)(x, deterministic=not train)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=not train)

        x = nn.Dense(features=10)(x)
        x = nn.softmax(x)

        return x
