#!/usr/bin/env python3
"""CPU-only Z-Image-Turbo pipeline for reference comparison."""
import sys
import torch
from diffusers import ZImagePipeline
from diffusers.utils import logging
logging.set_verbosity_error()

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
prompt = sys.argv[1] if len(sys.argv) > 1 else "A photo of a cat sitting on a windowsill"
output  = sys.argv[2] if len(sys.argv) > 2 else "out_cpu.png"
steps   = int(sys.argv[3]) if len(sys.argv) > 3 else 9
seed    = int(sys.argv[4]) if len(sys.argv) > 4 else 42

print(f"Prompt : {prompt!r}")
print(f"Steps  : {steps}  seed: {seed}  output: {output}")

pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
pipe = pipe.to("cpu")

generator = torch.Generator("cpu").manual_seed(seed)
image = pipe(
    prompt,
    height=512,
    width=512,
    num_inference_steps=steps,
    guidance_scale=0.0,   # required for Turbo models
    generator=generator,
).images[0]

image.save(output)
print(f"Saved → {output}")
