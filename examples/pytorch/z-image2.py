# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import ZImagePipeline
from diffusers.models.transformers import transformer_z_image
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift
from tt_torch import codegen_py

# CONFIG
compile_options = {
    "optimization_level": 1,
    "codegen_try_recover_structure": True,
}
EXPORT_PATH = "z_image_codegen"
torch_xla.set_custom_compile_options(compile_options)

MODEL_ID = "Tongyi-MAI/Z-Image"
DTYPE = torch.bfloat16
MODEL_CACHE_PATH = "z_image_pipeline.pt"


NUM_INFERENCE_STEPS = 1


def get_input_prompts():
    positive_prompt = "A photo of a cat sitting on a windowsill"
    negative_prompt = "Rain"

    return positive_prompt, negative_prompt


def get_input_latents(model):
    latents = model.prepare_latents(
        batch_size=1,
        num_channels_latents=model.transformer.in_channels,
        height=1280,
        width=720,
        dtype=DTYPE,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
    )

    return latents


def run_on_cpu_pipeline():
    print("\tRunning CPU pipeline...")

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

    image.save("example_cpu_pipeline.png")
    print("\t\tImage saved")

    return image


def main():
    run_on_cpu_pipeline()


if __name__ == "__main__":
    main()
