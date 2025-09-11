# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
from flax import linen as nn


class MNISTMLPModel(nn.Module):
    """MNIST MLP model implementation."""

    hidden_sizes: tuple[int]

    @nn.compact
    def __call__(self, x: jax.Array):
        x = x.reshape((x.shape[0], -1))

        for h in self.hidden_sizes:
            x = nn.Dense(features=h)(x)
            x = nn.relu(x)

        x = nn.Dense(features=10)(x)
        x = nn.softmax(x)

        return x
