# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Wan2.2-TI2V-5B-Turbo models.
"""

import torch
from diffusers import AutoencoderKLWan, GGUFQuantizationConfig, WanImageToVideoPipeline
from diffusers.models import WanTransformer3DModel
from huggingface_hub import hf_hub_download
from PIL import Image


def load_wan_ti2v_gguf_pipe(repo_id: str, gguf_filename: str, base_model: str):
    """Load a Wan TI2V pipeline with a GGUF-quantized transformer.

    Args:
        repo_id: HuggingFace repository ID containing the GGUF file.
        gguf_filename: Filename of the GGUF checkpoint within the repo.
        base_model: HuggingFace repository ID of the base Wan model for pipeline components.

    Returns:
        WanImageToVideoPipeline: Loaded pipeline with GGUF-quantized transformer.
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

    pipe = WanImageToVideoPipeline.from_pretrained(
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

    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        pipe.image_encoder.eval()
        for param in pipe.image_encoder.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def wan_ti2v_preprocessing(pipe, prompt, image=None):
    """Prepare inputs for the Wan TI2V pipeline.

    Args:
        pipe: WanImageToVideoPipeline instance.
        prompt: Text prompt for generation.
        image: Optional PIL image input. If None, a synthetic image is used.

    Returns:
        dict: Input arguments for the pipeline.
    """
    if image is None:
        image = Image.new("RGB", (832, 480), color=(128, 128, 200))

    return {
        "image": image,
        "prompt": prompt,
        "height": 480,
        "width": 832,
        "num_frames": 9,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
    }
