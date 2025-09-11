# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from flax import linen as nn


class MNISTCNNBatchNormModel(nn.Module):
    """MNIST CNN model implementation with batch normalization."""

    @nn.compact
    def __call__(self, x, *, train: bool):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=256)(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        x = nn.relu(x)

        x = nn.Dense(features=128)(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        x = nn.relu(x)

        x = nn.Dense(features=10)(x)
        x = nn.softmax(x)

        return x
