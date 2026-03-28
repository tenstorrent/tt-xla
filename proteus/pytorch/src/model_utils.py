# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Proteus model loading and processing.

Proteus is based on Stable Diffusion XL, so we reuse the SDXL preprocessing utilities.
"""

from diffusers import DiffusionPipeline
import torch


def load_pipe(variant):
    """Load Proteus pipeline.

    Args:
        variant: Model variant name

    Returns:
        DiffusionPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = DiffusionPipeline.from_pretrained(variant, torch_dtype=torch.float32)
    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]

    # Move the pipeline to CPU
    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe
