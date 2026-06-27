# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion 3 Medium text-to-image on Tenstorrent hardware.

Real-world scenario: a full text-to-image generation in which the MMDiT
transformer — the compute-dominant denoiser that runs once per diffusion step —
is compiled with the ``"tt"`` backend and executed on device, while the
(host-bound) T5/CLIP text encoders, the FlowMatch scheduler and the VAE run on
CPU. This mirrors the host-driven diffusion pattern in ``sd_v1_4_pipeline.py``
and ``sdxl-pipeline.py``, but instead of constructing the components by hand it
consumes the tt-forge-models SD3 loader: ``load_model`` returns the MMDiT
transformer and the loader caches the surrounding ``StableDiffusion3Pipeline``,
whose encoders / scheduler / VAE drive the rest of the loop.

The text encoders (T5-XXL dominates host RAM) are freed immediately after the
prompt is encoded and before the transformer is compiled onto the device — the
full pipeline otherwise OOMs a 32 GB host. Only the denoise-step count is
configurable for a faster smoke run; no model geometry (1024x1024, MMDiT block
count, sequence length) is reduced.
"""

import gc
from pathlib import Path

import torch
import torch_xla
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

# The loader's default prompt, so the example matches the bringup PoC.
PROMPT = ModelLoader.prompt
GUIDANCE_SCALE = 7.0
MAX_SEQUENCE_LENGTH = 256
SEED = 42


def post_process_output(image, output_path: str) -> str:
    """Save the generated image and print a human-readable result."""
    image.save(output_path)
    print(
        f'Prompt: "{PROMPT}"\n'
        f"Saved generated image to {output_path} "
        f"({image.width}x{image.height})"
    )
    return output_path


def run_sd3_medium(
    output_path: str = "sd3_medium_output.png", num_inference_steps: int = 28
) -> str:
    """Generate an image with SD3 Medium, running the MMDiT denoiser on TT.

    The MMDiT transformer is the only sub-module placed on the TT device; the
    text encoders and VAE stay on host. Host latents are hopped to the device for
    each transformer forward and back to CPU for the scheduler step, matching the
    per-step CPU<->TT casting in the SD 1.4 / SDXL examples.
    """
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()

    # Load in bf16: the full fp32 pipeline (~30 GB, T5-XXL dominates) OOMs a
    # 32 GB host before it can be downcast, so the dtype must reach from_pretrained.
    loader = ModelLoader(ModelVariant.STABLE_DIFFUSION_3_MEDIUM)
    transformer = loader.load_model(dtype_override=torch.bfloat16).eval()
    pipe = loader.pipeline  # the cached StableDiffusion3Pipeline (host components)

    height = width = pipe.default_sample_size * pipe.vae_scale_factor

    # 1. Encode the prompt on host (classifier-free guidance -> negative + positive).
    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=PROMPT,
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=True,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_prompt_embeds = torch.cat(
        [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
    )

    # 2. Free the text encoders before compiling the transformer on device — the
    #    T5-XXL encoder dominates host RAM and is no longer needed after encoding.
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.text_encoder_3 = None
    gc.collect()

    # 3. Compile the MMDiT denoiser and move it to the TT device.
    transformer.compile(backend="tt")
    transformer = transformer.to(device)

    # 4. Prepare latents and the dynamic-shift timesteps (FlowMatch scheduler).
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    latents = pipe.prepare_latents(
        1,
        transformer.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        "cpu",
        generator,
    )
    image_seq_len = (latents.shape[2] // transformer.config.patch_size) * (
        latents.shape[3] // transformer.config.patch_size
    )
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.16),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, "cpu", sigmas=None, mu=mu
    )

    # 5. Denoising loop — MMDiT forward on TT, guidance + scheduler step on host.
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)  # CFG: uncond + cond
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = transformer(
                hidden_states=latent_model_input.to(device),
                timestep=timestep.to(device),
                encoder_hidden_states=prompt_embeds.to(device),
                pooled_projections=pooled_prompt_embeds.to(device),
                return_dict=False,
            )[0].to("cpu")

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (
                noise_pred_text - noise_pred_uncond
            )
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            print(f"Step {i + 1}/{num_inference_steps}")

    # 6. VAE decode on host in fp32 for image quality.
    pipe.vae = pipe.vae.to(torch.float32)
    latents = (
        latents.to(torch.float32) / pipe.vae.config.scaling_factor
    ) + pipe.vae.config.shift_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return post_process_output(image, output_path)


def test_stable_diffusion_3_medium():
    """SD3 Medium produces a valid 1024x1024 image with the MMDiT on TT."""
    xr.set_device_type("TT")

    output_path = "test_sd3_medium_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    try:
        # A 2-step smoke run keeps the assertion cheap; geometry is unchanged.
        run_sd3_medium(output_path=output_path, num_inference_steps=2)

        assert output_file.exists(), f"Output image {output_path} was not created"
        with Image.open(output_path) as img:
            width, height = img.size
            assert (width, height) == (
                1024,
                1024,
            ), f"Expected 1024x1024, got {width}x{height}"
        print(f"Output image created with resolution {width}x{height}")
    finally:
        if output_file.exists():
            output_file.unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    xr.set_device_type("TT")
    run_sd3_medium()
