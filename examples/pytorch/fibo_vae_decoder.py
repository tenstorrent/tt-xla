# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
FIBO (briaai/FIBO) VAE-decoder example — Wan 2.2 VAE decode on a single TT device.

FIBO is BRIA AI's 8B-parameter DiT flow-matching text-to-image model. Its output
stage is the Wan 2.2 VAE (``AutoencoderKLWan``): after the DiT denoising loop
finishes, the scaled latents are decoded back to pixel space by ``vae.decode``.
This example isolates that **conv3d-heavy VAE decoder** and runs it end-to-end on
one Tenstorrent device, decoding a representative FIBO latent into a native
1024x1024 image.

The decoder is compiled at ``optimization_level=2``: level 2 applies the conv
memory-layout optimizations that give conv-based models their big performance win,
and (like every level >= 1) keeps GroupNorm as the native ``tenstorrent.group_norm``
kernel. The example loads weights/inputs through the tt-forge-models loader, runs
the decode on device, saves the reconstructed image, and asserts the output is a
finite, correctly-shaped, non-degenerate picture.

Notes:
- ``briaai/FIBO`` is a gated repo — accept the license on Hugging Face and set
  ``HF_TOKEN`` before running.
- 1024x1024 decode currently runs on p100 / p150 (Blackhole) TT devices.
"""

from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from PIL import Image

from third_party.tt_forge_models.fibo.vae_decoder.pytorch.loader import (
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    ModelLoader,
    ModelVariant,
)

# bf16 on device (the bringup baseline for this decoder); the loader draws a
# single deterministic fp32 latent and casts it to this dtype.
DTYPE = torch.bfloat16


def run_fibo_vae_decoder():
    """Decode a FIBO latent to a 1024x1024 image on a single TT device."""
    device = torch_xla.device()

    # optimization_level=2 — conv memory-layout optimizations (the drastic perf
    # win for conv-heavy models); GroupNorm stays the native composite kernel.
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    # Build the Wan 2.2 VAE decoder + a representative scaled latent via the loader.
    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model(dtype_override=DTYPE).eval()
    (latents,) = loader.load_inputs(dtype_override=DTYPE)

    # Move to device and compile for the TT backend.
    model = model.to(device)
    latents = latents.to(device)
    model.compile(backend="tt")

    with torch.no_grad():
        image = model(latents)

    return image.cpu()


def save_image(image: torch.Tensor, filepath: str) -> None:
    """Denormalize the VAE output ([-1, 1]) and save it as a PNG."""
    # Wan VAE decode returns [B, 3, T, H, W]; take the single frame -> [B, 3, H, W].
    if image.dim() == 5:
        image = image[:, :, 0, :, :]
    image = (torch.clamp(image.float() / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(
        torch.uint8
    )
    image_np = image.squeeze(0).cpu().numpy()  # [3, H, W]
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)  # [H, W, 3]
    Image.fromarray(image_np).save(filepath)


def post_process_output(image: torch.Tensor, output_path: str) -> None:
    """Save the decoded image and print a human-readable summary."""
    save_image(image, output_path)
    denorm = torch.clamp(image.float() / 2 + 0.5, 0.0, 1.0)
    print(f"FIBO VAE decoder output tensor: shape={tuple(image.shape)}")
    print(
        f"Decoded pixel stats (normalized [0,1]): "
        f"min={denorm.min():.4f} max={denorm.max():.4f} "
        f"mean={denorm.mean():.4f} std={denorm.std():.4f}"
    )
    print(f"Saved decoded {NATIVE_HEIGHT}x{NATIVE_WIDTH} image to {output_path}")


def test_fibo_vae_decoder():
    """Test the FIBO VAE decoder produces a finite, well-shaped, non-flat image."""
    xr.set_device_type("TT")

    image = run_fibo_vae_decoder()

    assert torch.isfinite(image.float()).all(), "VAE decoder output is not finite"

    expected_shape = (1, 3, 1, NATIVE_HEIGHT, NATIVE_WIDTH)
    assert (
        tuple(image.shape) == expected_shape
    ), f"Expected shape {expected_shape}, got {tuple(image.shape)}"

    # A real decoded image has spatial variation; a flat/garbage tensor would not.
    denorm = torch.clamp(image.float() / 2 + 0.5, 0.0, 1.0)
    assert denorm.std() > 1e-3, f"Decoded image is degenerate (std={denorm.std():.6f})"

    print(
        f"FIBO VAE decoder produced a valid {NATIVE_HEIGHT}x{NATIVE_WIDTH} image "
        f"(std={denorm.std():.4f})."
    )


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    output = run_fibo_vae_decoder()
    post_process_output(output, str(Path("fibo_vae_decoder_output.png")))
