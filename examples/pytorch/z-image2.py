# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import PIL.Image
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


class ZImageModule(nn.Module):
    """Wraps the full Z-Image pipeline into a single nn.Module.

    __init__ extracts all components from a ZImagePipeline.
    forward() runs the complete pipeline: text encoding, denoising loop with CFG,
    VAE decode, and postprocessing to a raw image tensor.
    """

    def __init__(self, pipe):
        super().__init__()
        self.text_encoder = pipe.text_encoder
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler

        self.vae_scaling_factor = pipe.vae.config.scaling_factor
        self.vae_shift_factor = pipe.vae.config.shift_factor

    def encode_prompt(self, prompt):
        """Encode a single prompt string, matching pipeline's _encode_prompt exactly."""
        chat_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        tokens = self.tokenizer(
            [chat_text],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask.bool()

        prompt_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]

        # Filter padding: return list of variable-length tensors (one per batch item)
        return [prompt_embeds[i][attention_mask[i]] for i in range(len(prompt_embeds))]

    def forward(
        self,
        positive_prompt,
        negative_prompt,
        latents,
        num_inference_steps,
        guidance_scale,
    ):
        """Run the full Z-Image pipeline.

        Args:
            positive_prompt: Positive prompt string.
            negative_prompt: Negative prompt string.
            latents: Initial noise tensor [B, 16, H, W].
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG guidance scale (>1 enables CFG).

        Returns:
            Image tensor [B, 3, H_img, W_img] in [0, 1] range.
        """
        # 1. Encode prompts
        prompt_embeds = self.encode_prompt(positive_prompt)
        negative_prompt_embeds = self.encode_prompt(negative_prompt)

        # 2. Compute timesteps with dynamic shifting
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        self.scheduler.set_timesteps(num_inference_steps, mu=mu)
        timesteps = self.scheduler.timesteps

        do_cfg = guidance_scale > 1

        # 3. Denoising loop
        for i, t in enumerate(timesteps):
            timestep = (1000 - t.expand(latents.shape[0])) / 1000

            if do_cfg:
                # CFG: double the batch
                latents_typed = latents.to(self.transformer.dtype)
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_input = prompt_embeds + negative_prompt_embeds
                timestep_input = timestep.repeat(2)
            else:
                latent_model_input = latents.to(self.transformer.dtype)
                prompt_embeds_input = prompt_embeds
                timestep_input = timestep

            # Prepare transformer input: [B, C, H, W] -> list of [C, 1, H, W]
            latent_model_input = latent_model_input.unsqueeze(2)
            latent_list = list(latent_model_input.unbind(dim=0))

            # Transformer forward
            model_out_list = self.transformer(
                latent_list, timestep_input, prompt_embeds_input, return_dict=False
            )[0]

            if do_cfg:
                # CFG combine
                actual_batch_size = latents.shape[0]
                pos_out = model_out_list[:actual_batch_size]
                neg_out = model_out_list[actual_batch_size:]

                noise_pred = []
                for j in range(actual_batch_size):
                    pos = pos_out[j].float()
                    neg = neg_out[j].float()
                    noise_pred.append(pos + guidance_scale * (pos - neg))
                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

            # Squeeze temporal dim + negate for flow matching
            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred

            # Scheduler step
            latents = self.scheduler.step(
                noise_pred.to(torch.float32), t, latents, return_dict=False
            )[0]

        # 4. VAE decode
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae_scaling_factor) + self.vae_shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        # 5. Postprocess: denormalize to [0, 1]
        image = (image * 0.5 + 0.5).clamp(0, 1)

        return image


def run_on_cpu_manual():
    print("\tRunning CPU manual...")

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

    image.save("example_cpu_manual.png")
    print("\t\tImage saved")

    return image


def main():
    run_on_cpu_manual()
    run_on_cpu_pipeline()


if __name__ == "__main__":
    main()
