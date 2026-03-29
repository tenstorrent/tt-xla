# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Wan2.2-TI2V-5B-Turbo models.
"""

import torch
from diffusers import AutoencoderKLWan, DiffusionPipeline, GGUFQuantizationConfig
from diffusers.models import WanTransformer3DModel
from huggingface_hub import hf_hub_download


def load_wan_ti2v_gguf_pipe(repo_id: str, gguf_filename: str, base_model: str):
    """Load a Wan TI2V pipeline with a GGUF-quantized transformer.

    Args:
        repo_id: HuggingFace repository ID containing the GGUF file.
        gguf_filename: Filename of the GGUF checkpoint within the repo.
        base_model: HuggingFace repository ID of the base Wan model for pipeline components.

    Returns:
        DiffusionPipeline: Loaded pipeline with GGUF-quantized transformer.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    transformer = WanTransformer3DModel.from_single_file(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    vae = AutoencoderKLWan.from_pretrained(
        base_model,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    pipe = DiffusionPipeline.from_pretrained(
        base_model,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    pipe.to("cpu")

    for module in [pipe.transformer, pipe.text_encoder, pipe.vae]:
        if module is not None:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    return pipe
