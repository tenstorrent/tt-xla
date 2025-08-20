# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for OFT Stable Diffusion model loading and processing.
"""

import peft.tuners.oft.layer as oft_layer
import torch
from diffusers import StableDiffusionPipeline
from peft import OFTConfig, OFTModel


def get_oft_configs():
    """Get OFT configurations for text encoder and UNet components.

    Returns:
        tuple: (config_te, config_unet) - OFT configurations for text encoder and UNet
    """
    config_te = OFTConfig(
        r=8,
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        module_dropout=0.0,
        init_weights=True,
    )
    config_unet = OFTConfig(
        r=8,
        target_modules=[
            "proj_in",
            "proj_out",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ],
        module_dropout=0.0,
        init_weights=True,
    )
    return config_te, config_unet


def patch_oft_cayley_with_lstsq():
    """Patch OFT Cayley method with least squares fallback for numerical stability."""

    def _safe_cayley(self, data):
        data = data.detach()
        b, r, c = data.shape
        skew_mat = 0.5 * (data - data.transpose(1, 2))
        id_mat = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)
        try:
            return torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat)
        except RuntimeError:
            return torch.linalg.lstsq(id_mat + skew_mat, id_mat - skew_mat).solution

    oft_layer.Linear._cayley_batch = _safe_cayley


def prepare_oft_pipeline(model_name: str):
    """Prepare Stable Diffusion pipeline with OFT adapters.

    Args:
        model_name: HuggingFace model name for Stable Diffusion pipeline

    Returns:
        StableDiffusionPipeline: Pipeline with OFT adapters applied
    """
    patch_oft_cayley_with_lstsq()
    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    config_te, config_unet = get_oft_configs()

    # Apply OFT adapters
    pipe.text_encoder = OFTModel(pipe.text_encoder, config_te, "default")
    pipe.unet = OFTModel(pipe.unet, config_unet, "default")

    # Disable gradient computation for OFT parameters
    for name, module in pipe.text_encoder.named_modules():
        if hasattr(module, "oft_r"):
            for key in module.oft_r:
                module.oft_r[key].requires_grad = False
        if hasattr(module, "oft_s"):
            for key in module.oft_r:
                module.oft_s[key].requires_grad = False

    for name, module in pipe.unet.named_modules():
        if hasattr(module, "oft_r"):
            for key in module.oft_r:
                module.oft_r[key].requires_grad = False
        if hasattr(module, "oft_s"):
            for key in module.oft_r:
                module.oft_s[key].requires_grad = False

    pipe.to("cpu")
    pipe.text_encoder.eval()
    pipe.unet.eval()

    return pipe


def generate_sample_inputs(
    pipe,
    prompt: str = "A beautiful mountain landscape during sunset",
    num_inference_steps: int = 30,
):
    """Generate sample inputs for the OFT Stable Diffusion model.

    Args:
        pipe: Prepared Stable Diffusion pipeline with OFT adapters
        prompt: Text prompt for generation
        num_inference_steps: Number of inference steps

    Returns:
        tuple: (latents, timestep, prompt_embeds) - Sample inputs for the model
    """
    # Encode prompt
    prompt_embeds, negative_prompt_embeds, *_ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt="",
        device="cpu",
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # Generate latents
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = height
    latents = torch.randn(
        (
            1,
            pipe.unet.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    latents = torch.cat([latents] * 2, dim=0)

    # Generate timestep
    pipe.scheduler.set_timesteps(num_inference_steps)
    timestep = pipe.scheduler.timesteps[0].expand(latents.shape[0])

    return (latents, timestep, prompt_embeds)
