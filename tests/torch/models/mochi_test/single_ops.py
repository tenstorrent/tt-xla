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

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


class CogVideoXCausalConv3d(nn.Module):
    """
    Minimal reproduction of CogVideoXCausalConv3d with replicate padding.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Padding: (width_left, width_right, height_top, height_bottom, time_front, time_back)
        self.time_causal_padding = (1, 1, 1, 1, 2, 0)

        # Conv3d with no padding (padding handled manually)
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=0
        )

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode="replicate")
        x = self.conv(x)
        return x


class ReplicatePadding(nn.Module):
    def __init__(self, time_causal_padding=(1, 1, 1, 1, 2, 0)):
        super().__init__()
        self.time_causal_padding = time_causal_padding

    def forward(self, x):
        return F.pad(x, self.time_causal_padding, mode="replicate")


class ReplicateConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=0
        )

    def forward(self, x):
        return self.conv(x)


def load_replicate_padding():
    return ReplicatePadding()


def load_conv3d():
    return ReplicateConv3d(in_channels=768, out_channels=768)


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


def load_first_resnet_block():
    """
    Load the first ResNet block from decoder.

    Tests both conv1 (✅ success with constant padding) and norm2 (❌ fails with reshape).
    This demonstrates two separate issues in one test:
    1. Constant padding fix works in conv1 layer
    2. Reshape error occurs in norm2 layer (separate compiler bug)

    Structure: decoder.block_in.resnets[0] which contains:
    - norm1
    - conv1 (CogVideoXCausalConv3d) <- padding patched here
    - norm2 <- fails with reshape error
    - conv2
    """
    print("Loading Mochi VAE decoder first ResNet block...")
    vae = AutoencoderKLMochi.from_pretrained(
        "genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.bfloat16
    )

    # Extract the first ResNet block from decoder
    resnet_block = vae.decoder.block_in.resnets[0]

    print(f"✓ Loaded ResNet block: {type(resnet_block).__name__}")
    print(f"  Contains: norm1, conv1, norm2, conv2")

    return resnet_block


def load_norm2_only():
    """
    Load ONLY the norm2 layer that fails with reshape error.

    Level 2 reproduction: Most isolated test of the failing normalization layer.
    This strips away the conv operations to focus purely on the reshape issue.

    Expected: Should fail at the same reshape operation as the full ResNet block.
    Input shape: [1, 768, 4, 60, 106] (matches conv1 output from full block)
    """
    print("Loading Mochi VAE decoder norm2 layer...")
    vae = AutoencoderKLMochi.from_pretrained(
        "genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.bfloat16
    )

    # Extract ONLY the norm2 layer that fails
    norm2 = vae.decoder.block_in.resnets[0].norm2

    print(f"✓ Loaded norm2 layer: {type(norm2).__name__}")
    print(f"  This layer performs: permute → reshape (fails on tile layout)")

    return norm2


def load_conv1_then_norm2():
    """
    Conv1 → Norm2 chain to reproduce reshape error.

    Level 3 reproduction: Minimal chain showing conv output → norm2 failure.
    This is smaller than full ResNet block but should still trigger the bug.

    Expected: Conv1 succeeds, then norm2 fails at reshape (same as full block)
    Purpose: Isolate the exact sequence that triggers context-dependent reshape bug
    """
    print("Loading Mochi VAE conv1 → norm2 chain...")
    vae = AutoencoderKLMochi.from_pretrained(
        "genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.bfloat16
    )

    # Extract both layers from first ResNet block
    resnet_block = vae.decoder.block_in.resnets[0]
    conv1 = resnet_block.conv1
    norm2 = resnet_block.norm2

    # Chain them together
    class Conv1ThenNorm2(nn.Module):
        def __init__(self, conv1, norm2):
            super().__init__()
            self.conv1 = conv1
            self.norm2 = norm2

        def forward(self, x):
            # Conv1 returns (output, cache) tuple - we only need the output
            conv_out = self.conv1(x)  # Should succeed with constant padding
            if isinstance(conv_out, tuple):
                x = conv_out[0]  # Extract output, discard cache
            else:
                x = conv_out
            x = self.norm2(x)  # Should fail at reshape
            return x

    model = Conv1ThenNorm2(conv1, norm2)
    print(f"✓ Created conv1 → norm2 chain")
    print(f"  Conv1 output will create problematic tile layout")
    print(f"  Norm2 reshape should fail on that layout")

    return model


def test_module(load_fn):
    xr.set_device_type("TT")
    device = torch_xla.device()

    print("\n" + "=" * 70)
    print(f"TEST: {load_fn.__name__}")
    print("=" * 70)

    mod = load_fn()
    mod = mod.to(torch.bfloat16).to(device).eval()
    mod = torch.compile(mod, backend="tt")

    # Exact Mochi dimensions from decoder: [B, C, T, H, W]
    # After VAE encoder: [1, 768, 4, 60, 106]
    x = torch.randn(1, 768, 4, 60, 106, dtype=torch.bfloat16).to(device)

    print(f"\nInput shape: {list(x.shape)}")
    print(f"Memory estimate: {x.numel() * 2 / 1024**2:.2f} MB")

    print(f"\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = mod(x)
        torch_xla.sync()
        print(f"✓ SUCCESS! Output shape: {output.shape}")
        return output
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        raise


if __name__ == "__main__":
    xr.set_device_type("TT")

    test_module(load_replicate_padding)
    # test_module(load_conv3d)
    # test_module(load_repro_conv1)
    # test_module(load_original_decoder_conv1)
    # test_module(load_first_resnet_block)
    # test_module(load_conv1_then_norm2) # Minimal repro of reshape error
    # test_module(load_norm2_only)
