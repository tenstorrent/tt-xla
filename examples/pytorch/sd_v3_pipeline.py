# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion 3 Medium text-to-image pipeline running end-to-end on TT.

The MMDiT transformer (the heavy net) runs on the Tenstorrent backend via
``torch.compile(backend="tt")``; the three text encoders (two CLIP + T5),
the scheduler and the VAE run on CPU. This mirrors the SD 1.5 / SDXL examples
in this directory: precision-sensitive text encoding and the VAE stay on CPU
while the dominant compute (the transformer) is offloaded to TT.

The denoising loop, prompt encoding, latent preparation and VAE decode reuse
the diffusers ``StableDiffusion3Pipeline`` helper methods directly, so the
numerics match upstream; only the per-step transformer call is redirected to
the TT device.
"""

from pathlib import Path
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    calculate_shift,
    retrieve_timesteps,
)


class SD3Config:
    def __init__(self, device="cpu"):
        self.model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        self.height = 1024
        self.width = 1024
        self.device = device


class SD3Pipeline:
    """Text-to-image generation with Stable Diffusion 3 Medium (MMDiT on TT)."""

    def __init__(self, config: SD3Config):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id
        self.height = config.height
        self.width = config.width

    def setup(self):
        # The text encoders, scheduler and VAE stay on CPU in fp32; only the
        # transformer is compiled for and moved to the TT device.
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        )
        self.pipe.to("cpu")

        self.transformer = self.pipe.transformer
        self.transformer = self.transformer.to(dtype=torch.bfloat16)
        self.transformer.compile(backend="tt")
        self.transformer = self.transformer.to(xm.xla_device())

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.0,
        num_inference_steps: int = 28,
        max_sequence_length: int = 256,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate an image from a text prompt. Returns tensor (B, 3, H, W) in [-1, 1]."""

        pipe = self.pipe
        do_cfg = guidance_scale > 1.0

        tt_cast = lambda x: (
            x.to(dtype=torch.bfloat16).to(device=xm.xla_device())
            if x.device == torch.device("cpu")
            else x.to(dtype=torch.bfloat16)
        )
        cpu_cast = lambda x: x.to("cpu").to(dtype=torch.float32)

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # --- Text encoding (CLIP x2 + T5) on CPU ---
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

            if do_cfg:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
                pooled_prompt_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                )

            # --- Prepare latents (CPU) ---
            num_channels_latents = pipe.transformer.config.in_channels
            latents = pipe.prepare_latents(
                1,
                num_channels_latents,
                self.height,
                self.width,
                prompt_embeds.dtype,
                self.device,
                generator,
                None,
            )

            # --- Prepare timesteps with dynamic shifting (CPU) ---
            scheduler_kwargs = {}
            if pipe.scheduler.config.get("use_dynamic_shifting", None):
                _, _, lat_h, lat_w = latents.shape
                image_seq_len = (lat_h // pipe.transformer.config.patch_size) * (
                    lat_w // pipe.transformer.config.patch_size
                )
                scheduler_kwargs["mu"] = calculate_shift(
                    image_seq_len,
                    pipe.scheduler.config.get("base_image_seq_len", 256),
                    pipe.scheduler.config.get("max_image_seq_len", 4096),
                    pipe.scheduler.config.get("base_shift", 0.5),
                    pipe.scheduler.config.get("max_shift", 1.16),
                )
            timesteps, num_inference_steps = retrieve_timesteps(
                pipe.scheduler,
                num_inference_steps,
                "cpu",
                sigmas=None,
                **scheduler_kwargs,
            )

            # --- Denoising loop (transformer on TT) ---
            for i, t in enumerate(timesteps):
                print(f"Step {i + 1} of {num_inference_steps}")

                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                timestep = t.expand(latent_model_input.shape[0])

                # CPU -> TT
                noise_pred = self.transformer(
                    hidden_states=tt_cast(latent_model_input),
                    timestep=tt_cast(timestep),
                    encoder_hidden_states=tt_cast(prompt_embeds),
                    pooled_projections=tt_cast(pooled_prompt_embeds),
                    return_dict=False,
                )[0]

                # TT -> CPU
                noise_pred = cpu_cast(noise_pred)

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = pipe.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

            # --- VAE decode (CPU) ---
            latents = (
                latents / pipe.vae.config.scaling_factor
            ) + pipe.vae.config.shift_factor
            latents = latents.to(dtype=torch.float32)
            images = pipe.vae.decode(latents, return_dict=False)[0]

            return images


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    """Rescale, reshape and save the image from pipeline output."""
    from PIL import Image

    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)


def run_sd3_pipeline(
    output_path: str = "output.png",
    prompt: str = "An astronaut riding a green horse",
    num_inference_steps: int = 28,
):
    """Run SD3 pipeline and save the output image."""
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    config = SD3Config(device="cpu")
    pipeline = SD3Pipeline(config=config)
    pipeline.setup()

    img = pipeline.generate(
        prompt=prompt,
        negative_prompt="",
        guidance_scale=7.0,
        num_inference_steps=num_inference_steps,
        seed=42,
    )

    save_image(img, output_path)
    return output_path


def test_sd3_pipeline():
    """Test SD3 pipeline generates a valid output image."""
    from PIL import Image

    xr.set_device_type("TT")

    output_path = "test_sd3_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_sd3_pipeline(output_path=output_path, num_inference_steps=28)

    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == 1024, f"Expected width 1024, got {width}"
        assert height == 1024, f"Expected height 1024, got {height}"

    print(f"Output image created with resolution {width}x{height}")
    print(f"Output image saved at {output_file.resolve()}")


if __name__ == "__main__":
    test_sd3_pipeline()
