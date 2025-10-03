# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from diffusers import StableDiffusion3Pipeline
import os


def test_sd():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        text_encoder_3=None,
        tokenizer_3=None,
        low_cpu_mem_usage=True,
        num_inference_steps=1,
    )
    # Tokenizer handling
    if hasattr(pipe, "tokenizer_2"):
        pipe.tokenizer_2.truncation = True
        pipe.tokenizer_2.padding = True

    pipe.transformer.forward = torch.compile(pipe.transformer.forward, backend="tt")

    prompt = "a photo of a cat holding a sign that says hello world"
    image = pipe(prompt).images[0]
    image.save("sd3_hello_world.png")
