# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


MIXER_TINY = {
    "image_size": 128,
    "patch_size": 16,
    "in_channels": 1,
    "dim": 128,
    "num_blocks": 4,
    "token_dim": 128,
    "channel_dim": 256,
    "num_classes": 32,
}


def create_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Extract non-overlapping patches on CPU into [B, num_patches, patch_dim]."""
    batch, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by patch size")

    patches_h = height // patch_size
    patches_w = width // patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(batch, patches_h * patches_w, channels * patch_size * patch_size)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArithmeticLayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(variance + self.eps)
        return centered * inv_std


class MixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_patches: int,
        token_dim: int,
        channel_dim: int,
    ):
        super().__init__()
        self.token_norm = ArithmeticLayerNorm()
        self.token_mix = FeedForward(num_patches, token_dim)
        self.channel_norm = ArithmeticLayerNorm()
        self.channel_mix = FeedForward(dim, channel_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mix(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.channel_norm(x)
        y = self.channel_mix(y)
        return x + y


class MLPMixer(nn.Module):
    def __init__(
        self,
        num_patches: int,
        patch_dim: int,
        dim: int,
        num_blocks: int,
        num_classes: int,
        token_dim: int,
        channel_dim: int,
    ):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    dim=dim,
                    num_patches=num_patches,
                    token_dim=token_dim,
                    channel_dim=channel_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = ArithmeticLayerNorm()
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def build_mixer(config: dict, dtype: torch.dtype = torch.bfloat16) -> MLPMixer:
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    in_channels = config["in_channels"]
    num_patches = (image_size // patch_size) ** 2
    patch_dim = in_channels * patch_size * patch_size

    model = MLPMixer(
        num_patches=num_patches,
        patch_dim=patch_dim,
        dim=config["dim"],
        num_blocks=config["num_blocks"],
        num_classes=config["num_classes"],
        token_dim=config["token_dim"],
        channel_dim=config["channel_dim"],
    )
    return model.to(dtype=dtype)
