# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr
from diffusers import AutoencoderKLMochi

os.environ["LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"


"""
Minimal reproduction of Conv3d operation that fails in Mochi decoder.

This isolates the exact Conv3d that causes:
  error: failed to legalize operation 'ttir.convolution' that was explicitly marked illegal

Reproduce the exact Conv3d from MochiDecoder3D.conv_in

From decoder.__init__():
    self.conv_in = nn.Conv3d(in_channels, block_out_channels[-1], kernel_size=(1, 1, 1))

Where:
    in_channels = 12 (latent channels)
    block_out_channels[-1] = 768
    kernel_size = (1, 1, 1) - point-wise convolution
"""


class MochiConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MochiConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv3d(x)


def test_conv3d_reproduction():
    device = torch_xla.device()

    # Create the Conv3d layer
    conv3d = MochiConv3d(in_channels=12, out_channels=768, kernel_size=(1, 1, 1))
    conv3d = conv3d.to(torch.bfloat16).to(device)

    conv3d = torch.compile(conv3d, backend="tt")

    # From decoder: [B, 12, t, h, w]
    # Example: [1, 12, 3, 32, 32] for 13 frames at 256x256
    batch = 1
    channels = 12
    temporal = 3
    height = 32
    width = 32

    x = torch.randn(batch, channels, temporal, height, width, dtype=torch.bfloat16)
    x = x.to(device)

    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = conv3d(x)
    print(f"✓ Success! Output shape: {output.shape}")
    print(f"Output: {output}")
    return output


def test_conv2d_reproduction():
    device = torch_xla.device()

    # Create the Conv2d layer
    conv2d = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=(1, 1))
    conv2d = conv2d.to(torch.bfloat16).to(device)

    conv2d = torch.compile(conv2d, backend="tt")
    batch = 1
    channels = 3
    height = 32
    width = 32

    x = torch.randn(batch, channels, height, width, dtype=torch.bfloat16)
    x = x.to(device)

    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = conv2d(x)
    print(f"✓ Success! Output shape: {output.shape}")
    return output


class CogVideoXCausalConv3d(nn.Module):
    """
    Minimal reproduction of CogVideoXCausalConv3d with replicate padding.

    Reproduces the exact error from Mochi decoder:
    "Statically allocated circular buffers grow to 9990400 B
     which is beyond max L1 size of 1499136 B"
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Padding: (width_left, width_right, height_top, height_bottom, time_front, time_back)
        self.time_causal_padding = (1, 1, 1, 1, 2, 0)

        # Conv3d with no padding (padding handled manually)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # Apply replicate padding (this causes gather/embedding lowering)
        x = F.pad(x, self.time_causal_padding, mode="replicate")
        return self.conv(x)


def load_original_decoder_conv1():
    """
    Load the actual conv1 layer from Mochi VAE decoder.

    This ensures we're testing the exact implementation used in the decoder,
    not a manual re-implementation that might differ.

    Returns:
        The conv1 CogVideoXCausalConv3d layer from the decoder
    """

    print("Loading Mochi VAE decoder...")
    vae = AutoencoderKLMochi.from_pretrained(
        "genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.bfloat16
    )

    # Extract conv1 from decoder
    # Decoder structure: decoder.conv1 is the first causal conv layer
    conv1 = vae.decoder.block_in.resnets[0].conv1

    print(f"✓ Loaded conv1: {type(conv1).__name__}")
    print(f"  Type: {type(conv1)}")
    print(f"  Conv sub-module: {type(conv1.conv) if hasattr(conv1, 'conv') else 'N/A'}")
    print(f"  Pad mode: {conv1.pad_mode if hasattr(conv1, 'pad_mode') else 'N/A'}")
    if hasattr(conv1, "conv") and hasattr(conv1.conv, "kernel_size"):
        print(f"  Kernel size: {conv1.conv.kernel_size}")

    return conv1

def load_repro_conv1():
    print("Loading repro conv1...")
    conv1 = CogVideoXCausalConv3d(in_channels=768, out_channels=768)

    print(f"✓ Loaded conv1: {type(conv1).__name__}")
    print(f"  Type: {type(conv1)}")
    print(f"  Conv sub-module: {type(conv1.conv) if hasattr(conv1, 'conv') else 'N/A'}")
    print(f"  Pad mode: {conv1.pad_mode if hasattr(conv1, 'pad_mode') else 'N/A'}")
    if hasattr(conv1, "conv") and hasattr(conv1.conv, "kernel_size"):
        print(f"  Kernel size: {conv1.conv.kernel_size}")

    return conv1


def test_causal_conv3d(load_fn):
    """
    Test the original CogVideoXCausalConv3d loaded from Mochi decoder.

    This uses the exact conv1 layer from the actual Mochi decoder,
    ensuring we're testing the real implementation, not a re-implementation.
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    print("\n" + "=" * 70)
    print(f"TEST: {load_fn.__name__}")
    print("=" * 70)

    # Load the actual conv1 from decoder
    conv1 = load_fn()
    conv1 = conv1.to(torch.bfloat16).to(device).eval()
    conv1 = torch.compile(conv1, backend="tt")

    # Exact Mochi dimensions from decoder: [B, C, T, H, W]
    # After VAE encoder: [1, 768, 4, 60, 106]
    x = torch.randn(1, 768, 4, 60, 106, dtype=torch.bfloat16).to(device)

    print(f"\nInput shape: {list(x.shape)}")
    print(f"Memory estimate: {x.numel() * 2 / 1024**2:.2f} MB")

    print(f"\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = conv1(x)
        torch_xla.sync()
        print(f"✓ SUCCESS! Output shape: {output.shape}")
        return output
    except Exception as e:
        print(f"✗ FAILED: {str(e)[:200]}")
        raise


if __name__ == "__main__":
    xr.set_device_type("TT")

    print("\n\n=== PART 1: Original CogVideoXCausalConv3d ===")
    try:
        test_causal_conv3d(load_original_decoder_conv1)
    except Exception:
        print("Original implementation test failed")
    
    print("\n" + "=" * 70)
    print("\n" + "=" * 70)
    print("\n\n=== PART 2: Repro CogVideoXCausalConv3d ===")
    
    try:
        test_causal_conv3d(load_repro_conv1)
    except Exception:
        print("Repro implementation test failed")