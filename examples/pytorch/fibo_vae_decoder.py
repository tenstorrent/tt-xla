# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
FIBO (briaai/FIBO) VAE-decoder single-chip inference example.

FIBO is BRIA AI's 8B DiT flow-matching text-to-image model. Its final pipeline
stage decodes the denoised latent to pixels with the Wan 2.2 3D causal VAE
(``diffusers.AutoencoderKLWan``, ``z_dim=48``, spatial stride 16). This example
brings up *only* that VAE decoder on a single chip — the DiT and SmolLM3 text
encoder are not loaded — which is the realistic standalone scenario for the
convolution-heavy decoder component.

A FIBO 1024x1024 image corresponds to a decoder-input latent of shape
``[1, 48, 1, 64, 64]`` (``B, z_dim, latent_frames, H//16, W//16``); the decoder
reconstructs a pixel tensor of shape ``[1, 3, 1, 1024, 1024]`` which is saved as
a PNG.

The decoder is compiled with ``optimization_level=2`` — the level where the
conv-based VAE's performance improves drastically on Tenstorrent hardware.
Weights and inputs come from the tt_forge_models FIBO loader; the input latent
is the loader's fixed-seed random latent (the DiT that would normally produce it
is out of scope here), so the reconstruction is an abstract image that exercises
the full decode path rather than a prompt-conditioned photo.

Reference: https://huggingface.co/briaai/FIBO
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from PIL import Image

from third_party.tt_forge_models.fibo.pytorch import ModelLoader, ModelVariant


def fibo_vae_decoder():
    """Decode a FIBO latent to pixels on a single TT device.

    Returns the reconstructed pixel tensor ``[1, 3, 1, 1024, 1024]`` on CPU.
    """
    # optimization_level=2 is where the conv-based VAE speeds up drastically.
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    # Weights + input latent via the tt_forge_models loader (public API only).
    loader = ModelLoader(ModelVariant.VAE_DECODER)
    model = loader.load_model().eval()
    latent = loader.load_inputs()[0]

    device = torch_xla.device()
    model = model.to(device=device)
    latent = latent.to(device=device)

    compiled = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled(latent)

    return output.cpu()


def post_process_output(output, filepath="fibo_vae_output.png"):
    """Save the decoded VAE frame as a PNG and print a human-readable summary."""
    # output: [B, 3, T, H, W] — take the single image frame.
    image = output.float()[0, :, 0]  # [3, H, W]
    image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(torch.uint8)
    image_np = image.permute(1, 2, 0).numpy()  # [H, W, 3]
    Image.fromarray(image_np).save(filepath)

    print(f"Decoded latent {tuple(output.shape)} -> image {image_np.shape[1]}x{image_np.shape[0]}")
    print(f"Pixel range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Saved decoded image to: {filepath}")


def test_fibo_vae_decoder():
    """Test the FIBO VAE decoder produces a finite, correctly-shaped image."""
    xr.set_device_type("TT")

    output = fibo_vae_decoder()

    assert output.shape == (1, 3, 1, 1024, 1024), f"Unexpected shape: {output.shape}"
    assert torch.isfinite(output.float()).all(), "Decoded output contains non-finite values"

    print(f"FIBO VAE decoder produced finite output of shape {tuple(output.shape)}.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    output = fibo_vae_decoder()
    post_process_output(output)
