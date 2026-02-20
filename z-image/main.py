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
EXPORT_DIR = DIR / "codegen"
torch_xla.set_custom_compile_options(compile_options)

MODEL_ID = "Tongyi-MAI/Z-Image"
DTYPE = torch.bfloat16

NUM_INFERENCE_STEPS = 1


def get_input_prompt():
    positive_prompt = "A photo of a cat sitting on a windowsill"

    return positive_prompt


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

    positive_prompt = get_input_prompt()
    latents = get_input_latents(pipe)

    print("\t\tGenerating image...")
    image = pipe(
        prompt=positive_prompt,
        negative_prompt="",
        height=1280,
        width=720,
        guidance_scale=1.0,
        cfg_normalization=False,
        num_inference_steps=NUM_INFERENCE_STEPS,
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
    model = ZImageModule(pipe, device="cpu")
    model.eval()
    print("\t\tModel loaded")

    positive_prompt = get_input_prompt()
    latents = get_input_latents(pipe)

    print("\t\tRunning forward...")
    with torch.no_grad():
        image_tensor = model(
            positive_prompt,
            latents,
            NUM_INFERENCE_STEPS,
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
    device = torch_xla.device()
    model = ZImageModule(pipe, device=device)
    model.eval()
    print("\t\tModel loaded")

    # Compile and move to TT device
    model.transformer.compile(backend="tt")
    model.text_encoder_module.compile(backend="tt")
    model.transformer = model.transformer.to(device)
    model.text_encoder_module = model.text_encoder_module.to(device)

    positive_prompt = get_input_prompt()
    latents = get_input_latents(pipe)

    print("\t\tRunning forward...")
    with torch.no_grad():
        image_tensor = model(
            positive_prompt,
            latents,
            NUM_INFERENCE_STEPS,
        )
    print("\t\tForward done")

    # Convert to PIL and save
    image_np = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    image_np = (image_np * 255).round().astype(np.uint8)
    image = PIL.Image.fromarray(image_np[0])

    image.save(DIR / "example_tt.png")
    print("\t\tImage saved")

    return image


def run_codegen(target="transformer"):
    """Run codegen on either the text_encoder or transformer.

    Args:
        target: "transformer" or "text_encoder"
    """
    assert target in ("transformer", "text_encoder")

    print(f"\n\tRunning codegen for {target}...")

    print("\t\tLoading pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
    )
    model = ZImageModule(pipe, device="cpu")
    model.eval()

    if target == "text_encoder":
        # Sample inputs: input_ids [1, 512], attention_mask [1, 512]
        tokens = model.tokenizer(
            [
                model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": get_input_prompt()}],
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
            export_path=str(EXPORT_DIR / "text_encoder"),
            compiler_options=compile_options,
        )

    elif target == "transformer":
        # Prepare sample transformer inputs as plain tensors
        latents = get_input_latents(pipe)
        prompt_embeds = model.encode_prompt(get_input_prompt())  # list of 1 tensor

        latent_input = latents.to(model.transformer.dtype).unsqueeze(2)[
            0
        ]  # [C, 1, H, W]
        timestep = torch.tensor([0.5])  # dummy normalized timestep
        cap_feat = prompt_embeds[0]  # [seq_len, 2560]

        # Apply graph-break-free forward now that we know the shapes
        import types

        from model import _make_transformer_forward

        new_fwd = _make_transformer_forward(
            model.transformer,
            cap_ori_len=cap_feat.shape[0],
            image_shape=tuple(latent_input.shape),
        )
        model.transformer.forward = types.MethodType(new_fwd, model.transformer)

        # Thin wrapper: codegen_py only moves plain tensor args to device
        class TransformerWrapper(torch.nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.transformer = transformer

            def forward(self, latent, t, cap):
                return self.transformer(
                    [latent],
                    t,
                    [cap],
                    return_dict=False,
                )[
                    0
                ][0]

        wrapper = TransformerWrapper(model.transformer)
        wrapper.eval()

        codegen_py(
            wrapper,
            latent_input,
            timestep,
            cap_feat,
            export_path=str(EXPORT_DIR / "transformer"),
            compiler_options=compile_options,
        )

    print(f"\t\tCodegen for {target} done")


def compare_images(label, a, b):
    a_arr = np.array(a, dtype=np.float32).flatten()
    b_arr = np.array(b, dtype=np.float32).flatten()
    pcc = np.corrcoef(a_arr, b_arr)[0, 1]
    diff = np.abs(a_arr - b_arr)
    bitwise = diff.max() == 0
    print(
        f"{label}:\n\tbitwise={bitwise}\n\tPCC={pcc:.10f}\n\tmax_diff={diff.max():.1f}/255\n\tmean_diff={diff.mean():.4f}/255"
    )


def main():
    out_golden = run_on_cpu_pipeline()
    out_cpu = run_on_cpu_manual()

    compare_images("Golden vs CPU", out_golden, out_cpu)

    # run_codegen(target="text_encoder")
    run_codegen(target="transformer")

    # out_tt = run_on_tt()


if __name__ == "__main__":
    main()
