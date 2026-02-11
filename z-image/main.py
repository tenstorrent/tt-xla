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
from tt_torch import codegen_py

from model import ZImageModule

DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# CONFIG
compile_options = {
    "optimization_level": 1,
    "codegen_try_recover_structure": False,
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

    print("")
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

    print("")
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

    print("")
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

    # Compile and move to TT device
    model.transformer.compile(backend="tt")
    model.text_encoder_module.compile(backend="tt")
    device = torch_xla.device()
    model.transformer = model.transformer.to(device)
    model.text_encoder_module = model.text_encoder_module.to(device)

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


def run_codegen(text_encoder=True, transformer=True):
    """Generate Python code for the Z-Image modules."""

    print("")
    print("\tGenerating codegen...")

    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
    )
    model = ZImageModule(pipe)
    model.eval()

    positive_prompt, negative_prompt = get_input_prompts()
    latents = get_input_latents(pipe)

    # --- Text encoder codegen ---
    if text_encoder:
        print("\t\tGenerating text encoder codegen...")
        tokens = model.tokenizer(
            [
                model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": positive_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            ],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        codegen_py(
            model.text_encoder_module,
            tokens.input_ids,
            tokens.attention_mask.bool(),
            export_path=EXPORT_PATH + "/text_encoder",
            compiler_options=compile_options,
        )
        print("\t\tText encoder codegen done")

    # --- Transformer codegen ---
    # The transformer takes List[Tensor] args, so wrap it to accept batched tensors.
    if transformer:

        class TransformerWrapper(torch.nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.transformer = transformer

            def forward(self, x, t, cap_feats):
                out_list = self.transformer(
                    list(x.unbind(dim=0)),
                    t,
                    list(cap_feats.unbind(dim=0)),
                    return_dict=False,
                )[0]
                return torch.stack(out_list)

        print("\t\tGenerating transformer codegen...")
        wrapper = TransformerWrapper(model.transformer)
        wrapper.eval()

        # Build sample inputs matching a single denoising step with CFG
        with torch.no_grad():
            prompt_embeds = model.encode_prompt(positive_prompt)
            negative_prompt_embeds = model.encode_prompt(negative_prompt)

        latents_typed = latents.to(model.transformer.dtype)
        latent_input = latents_typed.repeat(2, 1, 1, 1).unsqueeze(2)  # [2, C, 1, H, W]
        cap_feats_list = prompt_embeds + negative_prompt_embeds
        # Pad to same length and stack into a batched tensor
        max_len = max(c.shape[0] for c in cap_feats_list)
        cap_feats_padded = torch.stack(
            [
                torch.nn.functional.pad(c, (0, 0, 0, max_len - c.shape[0]))
                for c in cap_feats_list
            ]
        )  # [2, max_len, hidden_dim]
        timestep = torch.tensor([0.5, 0.5], dtype=torch.float32)

        codegen_py(
            wrapper,
            latent_input,
            timestep,
            cap_feats_padded,
            export_path=EXPORT_PATH + "/transformer",
            compiler_options=compile_options,
        )
        print("\t\tTransformer codegen done")

    print("\tCodegen complete")


def bitwise_compare(a, b):
    if (
        type(a) == PIL.PngImagePlugin.PngImageFile
        and type(b) == PIL.PngImagePlugin.PngImageFile
    ):
        return a == b
    elif type(a) == torch.Tensor and type(b) == torch.Tensor:
        return torch.equal(a, b)
    else:
        raise ValueError(f"Unsupported types: {type(a)} and {type(b)}")


def main():
    out_golden = run_on_cpu_pipeline()
    out_cpu = run_on_cpu_manual()

    print(f"Golden vs CPU: {bitwise_compare(out_golden, out_cpu)}")

    run_codegen(
        # text_encoder=True,
        text_encoder=False,
        transformer=True,
        # transformer=False,
    )

    out_tt = run_on_tt()


if __name__ == "__main__":
    main()
