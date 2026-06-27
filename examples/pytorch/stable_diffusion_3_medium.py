# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3 Medium text-to-image example.

This mirrors ``sd_v1_4_pipeline.py`` but drives the SD3 MMDiT transformer
through the tt-forge-models ``ModelLoader`` API. The heavy diffusion denoiser
(``SD3Transformer2DModel``) is compiled with the "tt" backend and runs on the
Tenstorrent device; the CLIP/T5 text encoders, the FlowMatch scheduler step,
and the VAE decode stay on host CPU. The denoise loop is a plain host-Python
loop that moves the four transformer inputs onto the device each step.

The example generates a full-resolution 1024x1024 image (latent 128x128) at the
model's native configuration -- no reduced geometry. SD3's scheduler has
``use_dynamic_shifting=False`` so no ``mu`` is needed for ``set_timesteps``.
"""
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from PIL import Image

from third_party.tt_forge_models.stable_diffusion_3.pytorch import (
    ModelLoader,
    ModelVariant,
)

# SD3-medium default guidance scale and inference step count.
GUIDANCE_SCALE = 7.0
NUM_INFERENCE_STEPS = 28


def generate_image(
    prompt: str,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    seed: int = 42,
) -> Image.Image:
    """Run the SD3-medium diffusion pipeline with the MMDiT denoiser on device.

    Returns a PIL image at the model's native 1024x1024 resolution.
    """
    device = torch_xla.device()

    # bf16: the full SD3 pipeline is ~30 GB in fp32 and OOM-kills a 31 GB host;
    # bf16 (~15 GB) fits and matches the bringup baseline. The loader materializes
    # the weights directly at the requested dtype.
    loader = ModelLoader(ModelVariant.STABLE_DIFFUSION_3_MEDIUM)
    transformer = loader.load_model(dtype_override=torch.bfloat16).eval()
    pipe = loader.pipeline  # full StableDiffusion3Pipeline (encoders, scheduler, VAE)

    tt_cast = lambda x: x.to(dtype=torch.bfloat16, device=device)
    cpu_cast = lambda x: x.to(device="cpu", dtype=torch.float32)

    with torch.no_grad():
        # --- Text encoding (CLIP x2 + T5) on CPU, with classifier-free guidance ---
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=True,
            device="cpu",
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

        # The text encoders are ~12 GB and no longer needed -- free them so the
        # host has room for the VAE decode at the end.
        pipe.text_encoder = None
        pipe.text_encoder_2 = None
        pipe.text_encoder_3 = None

        # --- Prepare timesteps (no mu: SD3 scheduler use_dynamic_shifting=False) ---
        scheduler = pipe.scheduler
        scheduler.set_timesteps(num_inference_steps, device="cpu")

        # --- Prepare native-resolution latents (1, 16, 128, 128) ---
        num_channels = transformer.config.in_channels
        height = width = pipe.default_sample_size * pipe.vae_scale_factor
        latent_h = height // pipe.vae_scale_factor
        latent_w = width // pipe.vae_scale_factor
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            (1, num_channels, latent_h, latent_w),
            generator=generator,
            dtype=torch.float32,
        )

        # --- Compile the MMDiT denoiser for the device ---
        transformer.compile(backend="tt")
        transformer = transformer.to(device)
        prompt_embeds_tt = tt_cast(prompt_embeds)
        pooled_prompt_embeds_tt = tt_cast(pooled_prompt_embeds)

        # --- Denoising loop (MMDiT on TT, scheduler step on CPU) ---
        for i, t in enumerate(scheduler.timesteps):
            # Classifier-free guidance: duplicate the latent for [uncond, cond].
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = transformer(
                hidden_states=tt_cast(latent_model_input),
                timestep=tt_cast(timestep),
                encoder_hidden_states=prompt_embeds_tt,
                pooled_projections=pooled_prompt_embeds_tt,
                return_dict=False,
            )[0]
            torch_xla.sync()
            noise_pred = cpu_cast(noise_pred)

            # CFG blend on CPU.
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample
            print(f"  step {i + 1}/{num_inference_steps} done")

        # --- VAE decode on CPU (fp32) ---
        vae = pipe.vae.to(dtype=torch.float32)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents.to(torch.float32)).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).round().to(torch.uint8).squeeze(0).permute(1, 2, 0)
    return Image.fromarray(image.cpu().numpy())


def post_process_output(image: Image.Image, output_path: str = "sd3_medium_output.png"):
    """Save the generated image and print a human-readable summary."""
    image.save(output_path)
    print(f"Generated image: {image.size[0]}x{image.size[1]}, saved to {output_path}")
    return output_path


def test_stable_diffusion_3_medium():
    """Smoke test: SD3-medium produces a finite native-resolution image."""
    xr.set_device_type("TT")

    output_path = "test_sd3_medium_output.png"
    out = Path(output_path)
    if out.exists():
        out.unlink()

    try:
        # Fewer steps than the default 28 to keep CI cheap -- native resolution
        # is preserved (only the number of denoise steps is reduced).
        image = generate_image(
            prompt="An astronaut riding a green horse",
            num_inference_steps=4,
        )
        assert image.size == (1024, 1024), f"Expected 1024x1024, got {image.size}"

        post_process_output(image, output_path)
        assert out.exists(), f"Output image {output_path} was not created"
    finally:
        if out.exists():
            out.unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    xr.set_device_type("TT")
    image = generate_image(
        prompt="An astronaut riding a green horse",
        num_inference_steps=NUM_INFERENCE_STEPS,
    )
    post_process_output(image)
