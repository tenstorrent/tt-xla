# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import ZImagePipeline

from model import ZImageModule

DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# CONFIG
compile_options = {
    "optimization_level": 1,
    "codegen_try_recover_structure": True,
}
EXPORT_PATH = "z_image_codegen"
torch_xla.set_custom_compile_options(compile_options)

MODEL_ID = "Tongyi-MAI/Z-Image"
DTYPE = torch.bfloat16

NUM_INFERENCE_STEPS = 1


def get_input_prompts():
    positive_prompt = "A photo of a cat sitting on a windowsill"
    negative_prompt = "Rain"

    return positive_prompt, negative_prompt


def get_input_latents(pipe):
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.in_channels,
        height=1280,
        width=720,
        dtype=DTYPE,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
    )

    return latents


def run_on_cpu_pipeline():
    image_path = DIR / "example_cpu_pipeline.png"

    print("\tRunning CPU pipeline...")

    # First check if output image exists
    # If it does, skip running and return the image
    if image_path.exists():
        print("\t\tOutput image already exists, skipping")
        return PIL.Image.open(image_path)

    print("\t\tLoading pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
    )
    print("\t\tPipeline loaded")

    positive_prompt, negative_prompt = get_input_prompts()
    latents = get_input_latents(pipe)

    print("\t\tGenerating image...")
    image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        height=1280,
        width=720,
        cfg_normalization=False,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=4,
        # generator=torch.Generator().manual_seed(42),
        latents=latents,
    ).images[0]
    print("\t\tImage generated")

    image.save(image_path)
    print("\t\tImage saved")

    return image


def run_on_cpu_manual():
    image_path = DIR / "example_cpu_manual.png"

    print("\tRunning CPU manual...")

    # First check if output image exists
    # If it does, skip running and return the image
    if image_path.exists():
        print("\t\tOutput image already exists, skipping")
        return PIL.Image.open(image_path)

    print("\t\tLoading pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
    )
    model = ZImageModule(pipe)
    model.eval()
    print("\t\tModel loaded")

    positive_prompt, negative_prompt = get_input_prompts()
    latents = get_input_latents(pipe)

    print("\t\tRunning forward...")
    with torch.no_grad():
        image_tensor = model(
            positive_prompt,
            negative_prompt,
            latents,
            NUM_INFERENCE_STEPS,
            guidance_scale=4,
        )
    print("\t\tForward done")

    # Convert to PIL and save
    image_np = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    image_np = (image_np * 255).round().astype(np.uint8)
    image = PIL.Image.fromarray(image_np[0])

    image.save(image_path)
    print("\t\tImage saved")

    return image


def run_on_tt():
    print("\tRunning on TT...")

    # Set up XLA runtime for TT backend
    xr.set_device_type("TT")

    print("\t\tLoading pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
    )
    model = ZImageModule(pipe)
    model.eval()
    print("\t\tModel loaded")

    # Compile transformer for TT and move to device
    model.transformer.compile(backend="tt")
    device = xm.xla_device()
    model.transformer = model.transformer.to(device)

    positive_prompt, negative_prompt = get_input_prompts()
    latents = get_input_latents(pipe)

    print("\t\tRunning forward...")
    with torch.no_grad():
        image_tensor = model(
            positive_prompt,
            negative_prompt,
            latents,
            NUM_INFERENCE_STEPS,
            guidance_scale=4,
        )
    print("\t\tForward done")

    # Convert to PIL and save
    image_np = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    image_np = (image_np * 255).round().astype(np.uint8)
    image = PIL.Image.fromarray(image_np[0])

    image.save(DIR / "example_tt.png")
    print("\t\tImage saved")

    return image


def bitwise_compare(a, b):
    if type(a) == PIL.Image.Image and type(b) == PIL.Image.Image:
        return a == b
    elif type(a) == torch.Tensor and type(b) == torch.Tensor:
        return torch.equal(a, b)
    else:
        raise ValueError(f"Unsupported types: {type(a)} and {type(b)}")


def main():
    out_golden = run_on_cpu_pipeline()
    out_cpu = run_on_cpu_manual()
    out_tt = run_on_tt()

    print(f"Golden vs CPU: {bitwise_compare(out_golden, out_cpu)}")


if __name__ == "__main__":
    main()
