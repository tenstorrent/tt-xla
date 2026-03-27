# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for RealVisXL model loading and processing.
"""

import torch
from diffusers import DiffusionPipeline


def load_pipe(variant):
    """Load RealVisXL pipeline.

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
