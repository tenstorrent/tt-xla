# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FLUX ControlNet Union model loading and processing.
"""

import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel


def load_flux_controlnet_union_pipe(controlnet_model_name, base_model_name):
    """Load FLUX ControlNet Union pipeline.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        base_model_name: Base FLUX model name on HuggingFace

    Returns:
        FluxControlNetPipeline: Loaded pipeline with components set to eval mode
    """
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model_name, torch_dtype=torch.bfloat16
    )
    pipe = FluxControlNetPipeline.from_pretrained(
        base_model_name, controlnet=controlnet, torch_dtype=torch.bfloat16
    )

    pipe.to("cpu")

    for component_name in [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "vae",
        "controlnet",
    ]:
        module = getattr(pipe, component_name, None)
        if module is not None:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    return pipe
