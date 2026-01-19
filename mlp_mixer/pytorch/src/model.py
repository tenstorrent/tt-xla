# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLP-Mixer model implementation for GitHub source
"""

import torch.nn as nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.0):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n d -> b d n"),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange("b d n -> b n d"),
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        image_size,
        channels,
        patch_size,
        dim,
        depth,
        num_classes,
        token_dim=256,
        channel_dim=2048,
        dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        patch_height, patch_width = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(
                MixerBlock(dim, num_patches, token_dim, channel_dim, dropout)
            )

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)

        return self.mlp_head(x)
