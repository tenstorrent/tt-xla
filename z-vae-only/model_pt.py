# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone PyTorch reference model for SD3.5 VAE decoder.

Self-contained — no imports from sibling directories.
Contains all building blocks for the VAE decoder from
stabilityai/stable-diffusion-3.5-large (via the Z-Image pipeline),
a VaeDecoderPT wrapper that handles loading from HuggingFace
or a local cache file, and a get_input() helper.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Tongyi-MAI/Z-Image"
DTYPE = torch.bfloat16
MODEL_CACHE_PATH = "vae_decoder.pt"

# VAE scaling parameters (SD3.5 VAE config)
SCALING_FACTOR = 1.5305
SHIFT_FACTOR = 0.0609

# Architecture constants
IN_CHANNELS = 16
OUT_CHANNELS = 3
BLOCK_OUT_CHANNELS = (128, 256, 512, 512)
LAYERS_PER_BLOCK = 2
NORM_NUM_GROUPS = 32
LATENT_CHANNELS = 16


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResNetBlock2D(nn.Module):
    """Residual block with GroupNorm + SiLU + Conv2d.

    Matches diffusers ResnetBlock2D with output_scale_factor=2.
    Weight names: norm1, conv1, norm2, conv2, conv_shortcut (optional).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut: Optional[nn.Conv2d] = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return x + residual


class AttentionBlock(nn.Module):
    """Self-attention block used in the VAE mid_block.

    Single head with head_dim = channels (512).
    Weight names: group_norm, to_q, to_k, to_v, to_out.0.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, C, H, W = x.shape

        x = self.group_norm(x)
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        scale = 1.0 / (self.channels**0.5)
        attn = torch.bmm(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        out = self.to_out[0](out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out + residual


class Upsample2D(nn.Module):
    """Nearest-neighbour 2x upsample followed by a 3x3 convolution.

    Weight names: conv.weight, conv.bias.
    """

    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class UNetMidBlock2D(nn.Module):
    """Mid-block: resnet[0] -> attention[0] -> resnet[1].

    Weight names:
        resnets.{0,1}.{norm1,conv1,norm2,conv2}.{weight,bias}
        attentions.0.{group_norm,to_q,to_k,to_v,to_out.0}.{weight,bias}
    """

    def __init__(
        self,
        channels: int,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResNetBlock2D(channels, channels, groups=groups, eps=eps),
                ResNetBlock2D(channels, channels, groups=groups, eps=eps),
            ]
        )
        self.attentions = nn.ModuleList(
            [AttentionBlock(channels, groups=groups, eps=eps)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class UpDecoderBlock2D(nn.Module):
    """Up-sampling decoder block: N resnets + optional upsampler.

    Weight names:
        resnets.{0..N-1}.{norm1,conv1,norm2,conv2}.{weight,bias}
        resnets.0.conv_shortcut.{weight,bias}  (when in_channels != out_channels)
        upsamplers.0.conv.{weight,bias}        (when add_upsample=True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        add_upsample: bool = True,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResNetBlock2D(res_in, out_channels, groups=groups, eps=eps)
            )
        self.upsamplers = nn.ModuleList()
        if add_upsample:
            self.upsamplers.append(Upsample2D(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """SD3.5 VAE Decoder.

    Architecture:
        conv_in -> mid_block -> up_blocks[0..3] -> conv_norm_out -> SiLU -> conv_out

    All weight names are prefixed with nothing (bare) so that when nested
    inside VaeDecoderPT the full path becomes e.g.
    ``decoder.conv_in.weight``, matching diffusers state-dict keys.
    """

    def __init__(
        self,
        groups: int = NORM_NUM_GROUPS,
        eps: float = 1e-6,
    ):
        super().__init__()

        # --- conv_in: latent_channels -> largest block channel ---
        self.conv_in = nn.Conv2d(LATENT_CHANNELS, 512, 3, padding=1)

        # --- mid block ---
        self.mid_block = UNetMidBlock2D(512, groups=groups, eps=eps)

        # --- up blocks (reversed channel order) ---
        # Block 0: 512 -> 512, 3 resnets, upsample
        # Block 1: 512 -> 512, 3 resnets, upsample
        # Block 2: 512 -> 256, 3 resnets (first has conv_shortcut), upsample
        # Block 3: 256 -> 128, 3 resnets (first has conv_shortcut), NO upsample
        # reversed_channels: [512, 512, 256, 128]
        # up_block[i] takes prev block's output channels as in_channels,
        # and reversed_channels[i] as out_channels.
        reversed_channels = list(reversed(BLOCK_OUT_CHANNELS))
        num_up_blocks = len(reversed_channels)
        self.up_blocks = nn.ModuleList()
        prev_out_ch = reversed_channels[0]  # 512 (output of mid_block)
        for i in range(num_up_blocks):
            out_ch = reversed_channels[i]
            add_upsample = i < num_up_blocks - 1
            self.up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=prev_out_ch,
                    out_channels=out_ch,
                    num_layers=LAYERS_PER_BLOCK + 1,  # 3 resnets per block
                    add_upsample=add_upsample,
                    groups=groups,
                    eps=eps,
                )
            )
            prev_out_ch = out_ch

        # --- output ---
        self.conv_norm_out = nn.GroupNorm(groups, 128, eps=eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(128, OUT_CHANNELS, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


# ---------------------------------------------------------------------------
# Wrapper: loading, scaling, and TTNN helpers
# ---------------------------------------------------------------------------


class VaeDecoderPT(nn.Module):
    """Wrapper around the VAE Decoder that handles weight loading and
    latent-space scaling.

    Loading priority:
        1. Local cache file (``cache_path``) — fastest.
        2. HuggingFace model hub via ``diffusers.DiffusionPipeline`` — will
           download the full Z-Image pipeline and extract the VAE decoder
           weights, then cache them locally.

    Forward pass applies VAE scaling before decoding::

        latent = latent / SCALING_FACTOR + SHIFT_FACTOR
        output = decoder(latent)
    """

    def __init__(
        self,
        cache_path: str = MODEL_CACHE_PATH,
        dtype: torch.dtype = DTYPE,
    ):
        super().__init__()
        self.decoder = Decoder()

        if os.path.exists(cache_path):
            state_dict = torch.load(cache_path, map_location="cpu", weights_only=True)
            self.decoder.load_state_dict(state_dict)
            print(f"[VaeDecoderPT] Loaded decoder weights from {cache_path}")
        else:
            self._load_from_hub(cache_path)

        self.decoder = self.decoder.to(dtype=dtype)
        self.decoder.eval()

    def _load_from_hub(self, cache_path: str) -> None:
        """Load VAE decoder weights from the Z-Image HuggingFace pipeline."""
        try:
            from diffusers import DiffusionPipeline
        except ImportError:
            raise RuntimeError(
                "diffusers is required to download the model. "
                "Install with: pip install diffusers"
            )

        print(f"[VaeDecoderPT] Downloading {MODEL_ID} (this may take a while)...")
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
        )

        # Extract decoder state dict from the pipeline VAE
        src_sd = pipe.vae.decoder.state_dict()
        missing, unexpected = self.decoder.load_state_dict(src_sd, strict=False)
        if missing:
            print(f"[VaeDecoderPT] Warning — missing keys: {missing}")
        if unexpected:
            print(f"[VaeDecoderPT] Warning — unexpected keys: {unexpected}")

        del pipe

        # Cache for future runs
        torch.save(self.decoder.state_dict(), cache_path)
        print(f"[VaeDecoderPT] Cached decoder weights to {cache_path}")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor, applying VAE scaling first."""
        latent = latent / SCALING_FACTOR + SHIFT_FACTOR
        return self.decoder(latent)

    def state_dict_for_ttnn(self) -> dict:
        """Return the decoder state dict with 'decoder.' prefix stripped.

        This is the bare decoder state dict (conv_in.weight, etc.)
        suitable for mapping to TTNN parameter names.
        """
        return self.decoder.state_dict()


# ---------------------------------------------------------------------------
# Input helper
# ---------------------------------------------------------------------------


def get_input(
    batch_size: int = 1,
    height: int = 160,
    width: int = 90,
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    """Return a sample latent tensor for the VAE decoder.

    Default shape is [1, 16, 160, 90] which corresponds to a
    1280x720 output image (8x spatial upscaling from the 4 up-blocks,
    but note only 3 have upsamplers so it is 2^3 = 8x).
    """
    return torch.randn(batch_size, LATENT_CHANNELS, height, width, dtype=dtype)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building VAE Decoder model...")
    model = VaeDecoderPT()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    latent = get_input()
    print(f"Input shape:  {latent.shape}")

    with torch.no_grad():
        output = model(latent)
    print(f"Output shape: {output.shape}")
