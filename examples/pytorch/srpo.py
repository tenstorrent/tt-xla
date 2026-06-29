# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Text-to-image generation with Tencent SRPO (a FLUX.1-dev fine-tune).

SRPO ships only the denoiser (the FLUX transformer); the CLIP/T5 text encoders,
VAE, tokenizers and scheduler come from black-forest-labs/FLUX.1-dev. The
denoiser is the compute-dominant component and the device target, so this
example compiles **only the SRPO transformer** with the "tt" backend and runs
the flow-matching denoising loop on the Tenstorrent device, while the text
encoders / scheduler / VAE stay on the host (mirrors sd_v1_4_pipeline.py and
wan's pipeline example).

The VAE decode is done on the host in fp32 with tiling enabled: a plain bf16
1024x1024 FLUX VAE decode is pathologically slow on CPU (a single 16k-token
self-attention), whereas fp32 + tiling keeps the decode tractable.

Model + components are obtained through the tt-forge-models loader API
(`third_party.tt_forge_models.srpo.pytorch.ModelLoader`); the full FLUX pipeline
the example drives is exposed on `loader.pipe`.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from PIL import Image

from third_party.tt_forge_models.srpo.pytorch import ModelLoader, ModelVariant

# Native generation geometry (published model card defaults, exposed by the loader).
HEIGHT = ModelLoader.NATIVE_HEIGHT
WIDTH = ModelLoader.NATIVE_WIDTH
MAX_SEQUENCE_LENGTH = ModelLoader.MAX_SEQUENCE_LENGTH
GUIDANCE_SCALE = ModelLoader.GUIDANCE_SCALE


def build_pipeline():
    """Load the SRPO FLUX pipeline via the loader and compile the denoiser for TT."""
    loader = ModelLoader(ModelVariant.BASE)
    # load_model() returns the SRPO transformer (== loader.pipe.transformer), bf16.
    loader.load_model()
    pipe = loader.pipe

    # Compile only the denoiser and move its weights to the device; the text
    # encoders, scheduler and VAE remain on the host.
    pipe.transformer.compile(backend="tt")
    pipe.transformer = pipe.transformer.to(xm.xla_device())
    return pipe


def generate(
    pipe,
    prompt: str,
    num_inference_steps: int,
    seed: Optional[int] = None,
    decode: bool = True,
) -> Union[Image.Image, torch.Tensor]:
    """Run the SRPO denoiser on device and (optionally) decode to an image.

    Returns a PIL image when ``decode`` is True, otherwise the raw denoised
    latents (used by the lightweight test to stay cheap).
    """
    device = xm.xla_device()
    dtype = pipe.transformer.dtype
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed if seed is not None else 0)

    with torch.no_grad():
        # --- Text encoding (host) ---
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device="cpu",
            num_images_per_prompt=1,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )

        # --- Latent preparation (host) ---
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, latent_image_ids = pipe.prepare_latents(
            1,
            num_channels_latents,
            HEIGHT,
            WIDTH,
            prompt_embeds.dtype,
            "cpu",
            generator,
            None,
        )

        # --- Flow-matching timestep schedule (FLUX dynamic shift) ---
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            latents.shape[1],
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            pipe.scheduler, num_inference_steps, "cpu", sigmas=sigmas, mu=mu
        )
        pipe.scheduler.set_begin_index(0)

        # FLUX.1-dev is guidance-distilled: an embedded guidance vector is always fed.
        guidance = torch.full([1], GUIDANCE_SCALE, dtype=dtype).expand(latents.shape[0])

        # Static transformer inputs: move to device once.
        guidance_dev = guidance.to(device)
        pooled_dev = pooled_prompt_embeds.to(device)
        prompt_embeds_dev = prompt_embeds.to(device)
        txt_ids_dev = text_ids.to(device)
        img_ids_dev = latent_image_ids.to(device)

        # --- Denoising loop (transformer on device, scheduler on host) ---
        for i, t in enumerate(timesteps):
            timestep = (t.expand(latents.shape[0]) / 1000).to(dtype).to(device)
            latents_dev = latents.to(device)

            noise_pred = pipe.transformer(
                hidden_states=latents_dev,
                timestep=timestep,
                guidance=guidance_dev,
                pooled_projections=pooled_dev,
                encoder_hidden_states=prompt_embeds_dev,
                txt_ids=txt_ids_dev,
                img_ids=img_ids_dev,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            xm.mark_step()

            noise_pred = noise_pred.to("cpu")
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            print(f"Step {i + 1}/{num_inference_steps} done")

        if not decode:
            return latents

        # --- VAE decode (host, fp32 + tiling to keep the 1024x1024 decode tractable) ---
        latents = pipe._unpack_latents(latents, HEIGHT, WIDTH, pipe.vae_scale_factor)
        latents = (
            latents / pipe.vae.config.scaling_factor
        ) + pipe.vae.config.shift_factor
        pipe.vae.to(torch.float32)
        pipe.vae.enable_tiling()
        image = pipe.vae.decode(latents.to(torch.float32), return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image


def post_process_output(image: Image.Image, output_path: str = "srpo_output.png"):
    """Save the generated image and print a human-readable summary."""
    image.save(output_path)
    print(f"Generated {image.width}x{image.height} image saved to: {output_path}")
    return output_path


def run_srpo(
    prompt: str = (
        "A majestic snow-capped mountain range at golden hour, a crystal-clear "
        "alpine lake in the foreground reflecting the peaks, ultra detailed, "
        "cinematic lighting"
    ),
    output_path: str = "srpo_output.png",
    num_inference_steps: int = 20,
    seed: int = 42,
):
    """Generate a single 1024x1024 image with SRPO and save it."""
    # Match the bringup baseline (bf16, optimization_level 0). On Blackhole the
    # optimization_level>=1 OpModel cost model opens a mock device whose worker
    # grid mismatches the registered system descriptor and aborts the compile.
    torch_xla.set_custom_compile_options({"optimization_level": 0})

    pipe = build_pipeline()
    image = generate(
        pipe, prompt, num_inference_steps=num_inference_steps, seed=seed, decode=True
    )
    return post_process_output(image, output_path)


def test_srpo():
    """Cheap guard: a few denoising steps at native geometry must stay finite."""
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 0})

    pipe = build_pipeline()
    latents = generate(
        pipe,
        "a photo of a cat",
        num_inference_steps=2,
        seed=42,
        decode=False,
    )

    assert torch.isfinite(latents).all(), "Denoised latents contain non-finite values"
    # FLUX packs the 1024x1024 latents into (B, (H/16)*(W/16), C*4) = (1, 4096, 64).
    assert latents.shape == (1, 4096, 64), f"Unexpected latent shape {latents.shape}"
    print(f"SRPO denoiser ran 2 steps; latent shape {tuple(latents.shape)}, all finite.")


if __name__ == "__main__":
    xr.set_device_type("TT")
    output_path = "srpo_output.png"
    if Path(output_path).exists():
        Path(output_path).unlink()
    run_srpo(output_path=output_path)
