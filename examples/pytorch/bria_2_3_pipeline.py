# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""BRIA 2.3 text-to-image pipeline running end-to-end on TT.

BRIA 2.3 (``briaai/BRIA-2.3``) is an SDXL-class text-to-image model: it reuses
the ``StableDiffusionXLPipeline``, the SDXL UNet and the SDXL preprocessing
path. As in the SDXL / SD 1.5 examples in this directory, the heavy net (the
UNet) runs on the Tenstorrent backend via ``torch.compile(backend="tt")`` while
the two CLIP text encoders, the scheduler and the VAE run on CPU.

The prompt encoding, latent preparation, denoising step and VAE decode reuse
the diffusers pipeline helper methods directly, so the numerics match upstream;
only the per-step UNet call is redirected to the TT device. The one BRIA-
specific tweak is ``force_zeros_for_empty_prompt = False`` (required per the
BRIA 2.3 model card).
"""

from pathlib import Path
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


class Bria23Config:
    def __init__(self, device="cpu"):
        self.model_id = "briaai/BRIA-2.3"
        self.height = 1024
        self.width = 1024
        self.device = device


class Bria23Pipeline:
    """Text-to-image generation with BRIA 2.3 (SDXL-class UNet on TT)."""

    def __init__(self, config: Bria23Config):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id
        self.height = config.height
        self.width = config.width

    def setup(self):
        # Text encoders, scheduler and VAE stay on CPU in fp32; only the UNet
        # is compiled for and moved to the TT device.
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        )
        # Required by BRIA 2.3: the empty prompt must not be zeroed out.
        self.pipe.force_zeros_for_empty_prompt = False
        self.pipe.to("cpu")

        self.unet = self.pipe.unet
        self.unet = self.unet.to(dtype=torch.bfloat16)
        self.unet.compile(backend="tt")
        self.unet = self.unet.to(xm.xla_device())

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate an image from a text prompt. Returns tensor (B, 3, H, W) in [-1, 1]."""

        pipe = self.pipe
        do_cfg = guidance_scale > 1.0
        crops_coords_top_left = (0, 0)
        original_size = target_size = (self.height, self.width)

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

            # --- Text encoding (CLIP x2) on CPU ---
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                device=self.device,
                num_images_per_prompt=1,
            )

            # --- Prepare timesteps (CPU) ---
            timesteps, num_inference_steps = retrieve_timesteps(
                pipe.scheduler, num_inference_steps, self.device
            )

            # --- Prepare latents (CPU) ---
            num_channels_latents = pipe.unet.config.in_channels
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

            # --- SDXL additional conditioning (time ids + pooled text embeds) ---
            if pipe.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

            add_time_ids = pipe._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            add_text_embeds = pooled_prompt_embeds

            if do_cfg:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
                add_text_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, add_text_embeds], dim=0
                )
                add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            # --- Denoising loop (UNet on TT) ---
            for i, t in enumerate(timesteps):
                print(f"Step {i + 1} of {num_inference_steps}")

                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # CPU -> TT
                noise_pred = self.unet(
                    tt_cast(latent_model_input),
                    tt_cast(t.unsqueeze(0)),
                    encoder_hidden_states=tt_cast(prompt_embeds),
                    added_cond_kwargs={
                        "text_embeds": tt_cast(add_text_embeds),
                        "time_ids": tt_cast(add_time_ids),
                    },
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
            has_latents_mean = (
                hasattr(pipe.vae.config, "latents_mean")
                and pipe.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(pipe.vae.config, "latents_std")
                and pipe.vae.config.latents_std is not None
            )
            latents = latents.to(dtype=torch.float32)
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(pipe.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / pipe.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / pipe.vae.config.scaling_factor

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


def run_bria_pipeline(
    output_path: str = "output.png",
    prompt: str = (
        "A portrait of a Beautiful and playful ethereal singer, "
        "golden designs, highly detailed, blurry background"
    ),
    num_inference_steps: int = 50,
):
    """Run BRIA 2.3 pipeline and save the output image."""
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    config = Bria23Config(device="cpu")
    pipeline = Bria23Pipeline(config=config)
    pipeline.setup()

    img = pipeline.generate(
        prompt=prompt,
        negative_prompt="",
        guidance_scale=5.0,
        num_inference_steps=num_inference_steps,
        seed=42,
    )

    save_image(img, output_path)
    return output_path


def test_bria_pipeline():
    """Test BRIA 2.3 pipeline generates a valid output image."""
    from PIL import Image

    xr.set_device_type("TT")

    output_path = "test_bria_2_3_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_bria_pipeline(output_path=output_path, num_inference_steps=50)

    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == 1024, f"Expected width 1024, got {width}"
        assert height == 1024, f"Expected height 1024, got {height}"

    print(f"Output image created with resolution {width}x{height}")
    print(f"Output image saved at {output_file.resolve()}")


if __name__ == "__main__":
    test_bria_pipeline()
