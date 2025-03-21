# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This file incorporates work covered by the following copyright and permission
# notice:
# SPDX-FileCopyrightText: Copyright 2024 Google LLC.
# SPDX-License-Identifier: Apache-2.0

# This code is based on google-research/vision_transformer

from typing import Any, Optional

import einops
import flax.linen as nn
import jax.numpy as jnp
import jax


class MlpBlock(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        y = nn.Dense(self.mlp_dim)(x)
        y = nn.gelu(y)
        return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
    """Mixer block layer."""

    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, name="token_mixing")(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y

        y = nn.LayerNorm()(x)
        y = MlpBlock(self.channels_mlp_dim, name="channel_mixing")(y)
        y = x + y

        return y


class MlpMixer(nn.Module):
    """Mixer architecture."""

    patches: Any
    num_classes: int
    num_blocks: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    model_name: Optional[str] = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = nn.Conv(
            self.hidden_dim, self.patches.size, strides=self.patches.size, name="stem"
        )(
            inputs
        )  # Patch embedding
        x = einops.rearrange(x, "n h w c -> n (h w) c")

        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)

        x = nn.LayerNorm(name="pre_head_layer_norm")(x)
        x = jnp.mean(x, axis=1)

        if self.num_classes:
            x = nn.Dense(
                self.num_classes, kernel_init=nn.initializers.zeros, name="head"
            )(x)

        return x
