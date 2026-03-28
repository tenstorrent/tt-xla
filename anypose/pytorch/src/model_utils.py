# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for AnyPose model loading and processing.
"""

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image


def load_anypose_pipe(base_model_name, lora_model_name, lora_scale=0.7):
    """Load AnyPose pipeline with LoRA weights.

    Args:
        base_model_name: Base Qwen Image Edit model name on HuggingFace
        lora_model_name: AnyPose LoRA model name on HuggingFace
        lora_scale: LoRA weight scale (default: 0.7)

    Returns:
        QwenImageEditPlusPipeline: Loaded pipeline with LoRA adapters
    """
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        base_model_name, torch_dtype=torch.float32
    )

    pipe.load_lora_weights(
        lora_model_name,
        weight_name="2511-AnyPose-base-000006250.safetensors",
        adapter_name="anypose_base",
    )
    pipe.load_lora_weights(
        lora_model_name,
        weight_name="2511-AnyPose-helper-00006000.safetensors",
        adapter_name="anypose_helper",
    )
    pipe.set_adapters(
        ["anypose_base", "anypose_helper"],
        adapter_weights=[lora_scale, lora_scale],
    )

    pipe.to("cpu")

    for component_name in ["text_encoder", "transformer", "vae"]:
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.eval()
            for param in component.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    return pipe


def create_dummy_images():
    """Create dummy input images for AnyPose inference.

    Returns:
        tuple: (character_image, pose_image) - Two PIL Images
    """
    character_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    pose_image = Image.new("RGB", (512, 512), color=(64, 64, 64))
    return character_image, pose_image


def anypose_preprocessing(pipe, prompt, character_image, pose_image):
    """Preprocess inputs for AnyPose model.

    Args:
        pipe: QwenImageEditPlusPipeline
        prompt: Text prompt for pose transfer
        character_image: PIL Image of the character to modify
        pose_image: PIL Image with the reference pose

    Returns:
        dict: Preprocessed inputs for the pipeline
    """
    inputs = {
        "image": [character_image, pose_image],
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    return inputs
