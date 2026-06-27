# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3 Medium (SD3) text-to-image example.

This mirrors the per-component diffusion examples in this directory
(``sd_v1_4_pipeline.py`` / ``sdxl-pipeline.py``): a host-Python denoising
loop that runs the heavy MMDiT denoiser (``SD3Transformer2DModel``) on the
Tenstorrent device via ``torch.compile(backend="tt")`` while the CLIP/T5 text
encoders, the FlowMatch scheduler and the VAE run on the CPU host.

Unlike SD 1.4 / SDXL, the model and the full pipeline (3 text encoders + VAE)
are obtained from the tt-forge-models loader API instead of being assembled
by hand here:

    from third_party.tt_forge_models.stable_diffusion_3.pytorch import (
        ModelLoader, ModelVariant,
    )
    loader      = ModelLoader(ModelVariant.STABLE_DIFFUSION_3_MEDIUM)
    transformer = loader.load_model(dtype_override=torch.bfloat16)  # MMDiT denoiser
    pipe        = loader.pipeline                                   # encoders/vae/scheduler

The pipeline is loaded directly in bf16 — materializing the full fp32 pipeline
(~30 GB, dominated by T5-XXL) OOMs a 32 GB host. The CLIP/T5 encoders are freed
right after prompt encoding so the host has headroom while the denoiser runs on
device. Images are generated at the model's native 1024x1024 resolution.
"""
import argparse
import gc
import time
from pathlib import Path
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from PIL import Image

from third_party.tt_forge_models.stable_diffusion_3.pytorch import (
    ModelLoader,
    ModelVariant,
)

# Native SD3 Medium geometry — never downscaled.
HEIGHT = WIDTH = 1024
GUIDANCE_SCALE = 7.0
MAX_SEQUENCE_LENGTH = 256


class SD3Pipeline:
    """Text-to-image pipeline for SD3 Medium with the MMDiT denoiser on TT."""

    def __init__(self):
        self.loader = ModelLoader(ModelVariant.STABLE_DIFFUSION_3_MEDIUM)
        self.transformer = None
        self.pipe = None
        self.vae = None
        self.scheduler = None

    def setup(self):
        # load_model loads the pipeline directly in bf16 and returns the
        # SD3Transformer2DModel (the TT-compilable denoiser).
        self.transformer = self.loader.load_model(dtype_override=torch.bfloat16).eval()
        self.pipe = self.loader.pipeline
        self.scheduler = self.pipe.scheduler
        # VAE decode runs on the CPU host in fp32 for numerical stability.
        self.vae = self.pipe.vae.to(torch.float32)

        # Compile the denoiser for the TT backend and move it to the device.
        self.transformer.compile(backend="tt")
        self.transformer = self.transformer.to(xm.xla_device())

    def _encode_prompt(self, prompt: str, negative_prompt: str):
        """Encode the prompt with the (CPU) CLIP-L/CLIP-G/T5 encoders, then free them."""
        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
                negative_prompt_3=None,
                do_classifier_free_guidance=True,
                device="cpu",
                num_images_per_prompt=1,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )

        # Classifier-free guidance: stack [uncond, cond].
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

        # Free the (large) text encoders — T5-XXL alone is ~9.5 GB bf16 — so the
        # host has headroom while the denoiser runs and the VAE decodes.
        self.pipe.text_encoder = None
        self.pipe.text_encoder_2 = None
        self.pipe.text_encoder_3 = None
        gc.collect()

        return prompt_embeds, pooled_prompt_embeds

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """Generate an image from a text prompt. Returns a (B, 3, H, W) tensor."""
        device = xm.xla_device()
        tt_cast = lambda x: x.to(dtype=torch.bfloat16).to(device=device)
        cpu_cast = lambda x: x.to("cpu").to(dtype=torch.float32)

        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
            prompt, negative_prompt
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed if seed is not None else generator.seed())

        # Latents at native resolution: (1, 16, 128, 128).
        num_channels_latents = self.transformer.config.in_channels
        latents = torch.randn(
            (1, num_channels_latents, HEIGHT // 8, WIDTH // 8),
            generator=generator,
            dtype=torch.float32,
        )

        self.scheduler.set_timesteps(num_inference_steps, device="cpu")

        start = time.time()
        with torch.no_grad():
            for i, timestep in enumerate(self.scheduler.timesteps):
                print(f"Step {i + 1}/{num_inference_steps}")
                # CFG: duplicate latents into [uncond, cond] batch.
                latent_model_input = torch.cat([latents] * 2)
                ts = timestep.expand(latent_model_input.shape[0])

                # CPU -> TT
                noise_pred = self.transformer(
                    hidden_states=tt_cast(latent_model_input),
                    timestep=tt_cast(ts),
                    encoder_hidden_states=tt_cast(prompt_embeds),
                    pooled_projections=tt_cast(pooled_prompt_embeds),
                    return_dict=False,
                )[0]
                torch_xla.sync()

                # TT -> CPU, then classifier-free guidance blend.
                noise_pred = cpu_cast(noise_pred)
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)

                # FlowMatch scheduler step (CPU).
                latents = self.scheduler.step(
                    noise_pred, timestep, latents, return_dict=False
                )[0]
        print(f"Denoising time: {time.time() - start:.1f}s")

        # VAE decode (CPU, fp32).
        start = time.time()
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents.to(torch.float32), return_dict=False)[0]
        print(f"VAE decode time: {time.time() - start:.1f}s")
        return image


def save_image(image: torch.Tensor, filepath: str = "sd3_output.png"):
    """Rescale, reshape and save the image tensor from the pipeline output."""
    image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(torch.uint8)
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    Image.fromarray(image_np).save(filepath)
    return filepath


def post_process_output(image: torch.Tensor, prompt: str, output_path: str):
    """Print a human-readable result for the generated image."""
    path = save_image(image, output_path)
    _, _, h, w = image.shape
    print("\n=== Stable Diffusion 3 Medium ===")
    print(f'Prompt           : "{prompt}"')
    print(f"Image resolution : {w}x{h}")
    print(f"Saved image to   : {Path(path).resolve()}")


def run_sd3_pipeline(
    prompt: str = "An astronaut riding a green horse",
    output_path: str = "sd3_output.png",
    num_inference_steps: int = 28,
) -> torch.Tensor:
    """Run the SD3 Medium pipeline and return the generated image tensor."""
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    pipeline = SD3Pipeline()
    pipeline.setup()
    return pipeline.generate(prompt, num_inference_steps=num_inference_steps)


def test_stable_diffusion_3_medium():
    """Smoke test: SD3 Medium denoiser runs on device and yields a valid image."""
    xr.set_device_type("TT")

    # Native 1024x1024 geometry; a small step count keeps CI cheap.
    image = run_sd3_pipeline(num_inference_steps=2)

    assert image is not None, "Pipeline returned no image"
    assert image.shape == (1, 3, HEIGHT, WIDTH), f"Unexpected shape {image.shape}"
    assert torch.isfinite(image).all(), "Image contains non-finite values"
    print(f"test passed: finite image with shape {tuple(image.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD3 Medium text-to-image on TT")
    parser.add_argument(
        "--prompt", type=str, default="An astronaut riding a green horse"
    )
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--output_path", type=str, default="sd3_output.png")
    args = parser.parse_args()

    xr.set_device_type("TT")

    image = run_sd3_pipeline(
        prompt=args.prompt,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
    )
    post_process_output(image, args.prompt, args.output_path)
