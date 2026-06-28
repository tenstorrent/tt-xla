# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion 3 Medium text-to-image on Tenstorrent hardware.

End-to-end text-to-image generation with SD3 Medium. The compute-heavy MMDiT
transformer (``SD3Transformer2DModel``) is compiled with ``backend="tt"`` and
runs on the TT device; the text encoders, FlowMatch scheduler and VAE stay on
host (the standard diffusion split, mirroring ``sdxl-pipeline.py`` and
``sd_v1_4_pipeline.py``).

The model, prompt and host components all come from the tt-forge-models loader
(``stable_diffusion_3``); this script only wraps them in a hand-rolled denoise
loop so the denoiser can hop CPU<->TT each step. The text encoders are freed
right after prompt encoding — SD3 Medium's full pipeline is ~15 GB at bf16 and
the host RAM wall (~31 GB) is the binding constraint, so the encoders must be
released before the transformer compiles and moves to device.
"""
import gc
from pathlib import Path

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    calculate_shift,
    retrieve_timesteps,
)
from PIL import Image

from third_party.tt_forge_models.stable_diffusion_3.pytorch import (
    ModelLoader,
    ModelVariant,
)

# bf16 on device, fp16 for host host<->device transfers, fp32 only for the VAE
# decode (bf16 conv decode at 1024^2 on CPU is pathological).
tt_cast = lambda x: x.to(dtype=torch.bfloat16, device=xm.xla_device())
cpu_cast = lambda x: x.to(device="cpu", dtype=torch.float16)


class SD3MediumGenerator:
    """Hand-rolled SD3 Medium text-to-image driver (MMDiT on TT, rest on host)."""

    def __init__(self, num_inference_steps: int = 28, guidance_scale: float = 7.0):
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.loader = ModelLoader(ModelVariant.STABLE_DIFFUSION_3_MEDIUM)

    def setup(self):
        """Load the transformer (bf16) + host pipeline components from the loader."""
        # load_model populates loader.pipeline as a side effect; reach the host
        # components (encoders / scheduler / VAE) through that public attribute.
        self.transformer = self.loader.load_model(dtype_override=torch.bfloat16).eval()
        self.pipe = self.loader.pipeline
        self.prompt = self.loader.prompt
        # Native SD3 Medium geometry — 1024x1024.
        self.height = self.pipe.default_sample_size * self.pipe.vae_scale_factor
        self.width = self.pipe.default_sample_size * self.pipe.vae_scale_factor

    def _encode_then_free_encoders(self, prompt: str, negative_prompt: str):
        """Encode the prompt on host, then free the three text encoders."""
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
            do_classifier_free_guidance=True,
            device="cpu",
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
        # CFG: stack [uncond, cond] along the batch dim.
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

        # Encoders are only needed to build the embeddings — release ~9.5 GB of
        # T5-XXL + CLIP weights before compiling/moving the transformer to device.
        self.pipe.text_encoder = None
        self.pipe.text_encoder_2 = None
        self.pipe.text_encoder_3 = None
        gc.collect()

        return prompt_embeds, pooled_prompt_embeds

    def generate(self, prompt: str = None, negative_prompt: str = "", seed: int = 42):
        """Run the full text-to-image generation. Returns an image tensor (B,3,H,W)."""
        prompt = prompt or self.prompt
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_then_free_encoders(
                prompt, negative_prompt
            )

            # --- Latents (host) ---
            generator = torch.Generator(device="cpu").manual_seed(seed)
            num_channels_latents = self.transformer.config.in_channels
            latents = torch.randn(
                (
                    1,
                    num_channels_latents,
                    int(self.height) // self.pipe.vae_scale_factor,
                    int(self.width) // self.pipe.vae_scale_factor,
                ),
                generator=generator,
                dtype=torch.float32,
            )

            # --- Timesteps with SD3 dynamic shifting (host) ---
            scheduler = self.pipe.scheduler
            image_seq_len = (
                latents.shape[2] // self.transformer.config.patch_size
            ) * (latents.shape[3] // self.transformer.config.patch_size)
            mu = calculate_shift(
                image_seq_len,
                scheduler.config.base_image_seq_len,
                scheduler.config.max_image_seq_len,
                scheduler.config.base_shift,
                scheduler.config.max_shift,
            )
            timesteps, _ = retrieve_timesteps(
                scheduler, self.num_inference_steps, "cpu", sigmas=None, mu=mu
            )

            # --- Compile + move the denoiser to the TT device ---
            self.transformer.compile(backend="tt")
            self.transformer = self.transformer.to(xm.xla_device())
            prompt_embeds = tt_cast(prompt_embeds)
            pooled_prompt_embeds = tt_cast(pooled_prompt_embeds)

            # --- Denoising loop (MMDiT on TT, CFG + scheduler on host) ---
            for i, t in enumerate(timesteps):
                print(f"Step {i + 1}/{self.num_inference_steps}")
                # CFG: duplicate latents (flow-matching scheduler does not scale).
                latent_model_input = torch.cat([latents] * 2)
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=tt_cast(latent_model_input),
                    timestep=tt_cast(timestep),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                noise_pred = cpu_cast(noise_pred).to(torch.float32)

                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + self.guidance_scale * (
                    noise_cond - noise_uncond
                )
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # --- VAE decode (host, fp32) ---
            vae = self.pipe.vae.float()
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents.float(), return_dict=False)[0]
            return image


def post_process_output(image: torch.Tensor, output_path: str = "sd3_medium_output.png"):
    """Save the decoded image and print where it landed (human-readable result)."""
    image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(torch.uint8)
    image_np = image.detach().cpu().squeeze().numpy()
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
    Image.fromarray(image_np).save(output_path)
    print(f"Saved generated image to: {output_path} ({image_np.shape[1]}x{image_np.shape[0]})")
    return output_path


def run_sd3_medium(output_path: str = "sd3_medium_output.png", num_inference_steps: int = 28):
    """Generate an image with SD3 Medium and save it. Returns (path, image tensor)."""
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    generator = SD3MediumGenerator(num_inference_steps=num_inference_steps)
    generator.setup()
    image = generator.generate()
    post_process_output(image, output_path)
    return output_path, image


def test_stable_diffusion_3_medium():
    """SD3 Medium runs on device and produces a finite, native-resolution image."""
    xr.set_device_type("TT")

    output_path = "test_sd3_medium_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()
    try:
        # 2 steps keeps CI cheap while still exercising compile + the full loop.
        _, image = run_sd3_medium(output_path=output_path, num_inference_steps=2)

        assert torch.isfinite(image).all(), "Decoded image contains non-finite values"
        assert output_file.exists(), f"Output image {output_path} was not created"
        with Image.open(output_path) as img:
            assert img.size == (1024, 1024), f"Expected 1024x1024, got {img.size}"
        print("test_stable_diffusion_3_medium passed")
    finally:
        if output_file.exists():
            output_file.unlink()


if __name__ == "__main__":
    xr.set_device_type("TT")
    run_sd3_medium()
