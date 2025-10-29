# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import MochiPipeline

# Enable HLO debug output
os.environ["XLA_HLO_DEBUG"] = "1"


def run_transformer():
    """
    Test Mochi DiT (Diffusion Transformer) in isolation.

    Input:
    - Noisy latents [B, 12, t, h, w]
    - Sigma (noise level) [B]
    - Text embeddings [B, 256, 4096] from T5-XXL
    - Text mask [B, 256]

    Output: Predicted velocity [B, 12, t, h, w]

    Mochi DiT specs:
    - AsymmDiT architecture: 10B parameters
    - 48 layers, 24 heads, visual_dim=3072, text_dim=1536
    """
    xr.set_device_type("TT")

    # Load Mochi pipeline to get transformer
    pipeline = MochiPipeline.from_pretrained(
        "genmo/mochi-1-preview", torch_dtype=torch.float16, variant="bf16"
    )

    # Prepare inputs
    # Noisy latents: [1, 12, 3, 32, 32]
    latent = torch.randn(1, 12, 3, 32, 32, dtype=torch.bfloat16)

    # Noise level sigma: [1], range [0, 1]
    sigma = torch.tensor([0.5], dtype=torch.bfloat16)

    # Text features from T5-XXL: [1, 256, 4096]
    # In practice, this comes from T5 encoder, here we use dummy
    text_features = torch.randn(1, 256, 4096, dtype=torch.bfloat16)

    # Text attention mask: [1, 256]
    # 1 = valid token, 0 = padding
    text_mask = torch.ones(1, 256, dtype=torch.bool)
    # Simulate some padding (only first 128 tokens are real)
    text_mask[:, 128:] = False

    model = pipeline.transformer
    model = model.to(torch.bfloat16)
    model = model.eval()

    # Compile with TT backend
    model.compile(backend="tt")

    device = xm.xla_device()

    latent = latent.to(device)
    sigma = sigma.to(device)
    text_features = text_features.to(device)
    text_mask = text_mask.to(device)
    model = model.to(device)

    with torch.no_grad():
        # DiT forward pass
        output = model(latent, sigma, text_features, text_mask)

    print(f"Transformer output shape: {output.shape}")
    print(f"Expected shape: [1, 12, 3, 32, 32] (same as input latent)")
    print(output)


if __name__ == "__main__":
    print("Running Mochi DiT test...")
    run_transformer()
