# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from diffusers.models.transformers import transformer_z_image
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


class TextEncoderModule(nn.Module):
    """Wraps Qwen3 to run all decoder layers directly, avoiding the
    hook-based output_hidden_states mechanism that breaks torch.compile.

    Returns the output of the last decoder layer (before final RMSNorm),
    which is equivalent to
    text_encoder(..., output_hidden_states=True).hidden_states[-2].

    Why [-2] == last layer output: HuggingFace collects hidden_states as
    [input_embeds, layer_0_out, ..., layer_{N-1}_out] then replaces the
    last entry with the normed last_hidden_state, so [-2] is layer N-1's
    raw output.
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.embed_tokens = text_encoder.embed_tokens
        self.rotary_emb = text_encoder.rotary_emb
        num_layers = text_encoder.config.num_hidden_layers
        self.layers = text_encoder.layers[: num_layers - 1]
        self.config = text_encoder.config
        self.has_sliding_layers = text_encoder.has_sliding_layers

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embed_tokens(input_ids)

        cache_position = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = (
                create_sliding_window_causal_mask(**mask_kwargs)
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )

        return hidden_states


class ZImageModule(nn.Module):
    """Wraps the full Z-Image pipeline into a single nn.Module.

    __init__ extracts all components from a ZImagePipeline and applies
    the RoPE monkey-patch (complex64 -> real-valued) for XLA compatibility.

    forward() runs the complete pipeline: text encoding, denoising loop with CFG,
    VAE decode, and postprocessing to a raw image tensor. The transformer inputs
    are automatically moved to/from whatever device it lives on.
    """

    def __init__(self, pipe, device):
        super().__init__()
        self.text_encoder_module = TextEncoderModule(pipe.text_encoder)
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
        self.vae_scaling_factor = pipe.vae.config.scaling_factor
        self.vae_shift_factor = pipe.vae.config.shift_factor

        self.device = device

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

        print(f"input_ids.shape: {input_ids.shape}")
        print(f"attention_mask.shape: {attention_mask.shape}")

        # Move inputs to same device as text encoder
        te_device = self.text_encoder_module.embed_tokens.weight.device
        prompt_embeds = self.text_encoder_module(
            input_ids.to(te_device),
            attention_mask.to(te_device),
        )

        print(f"prompt_embeds.shape: {prompt_embeds.shape}")
        # Filter padding on CPU (variable-length output, can't be compiled)
        prompt_embeds = prompt_embeds.cpu()
        attention_mask = attention_mask.cpu()
        return [prompt_embeds[i][attention_mask[i]] for i in range(len(prompt_embeds))]

    def forward(self, prompt, latents, num_inference_steps):
        """Run the full Z-Image pipeline (single prompt, no CFG).

        Args:
            prompt: Positive prompt string.
            latents: Initial noise tensor [1, 16, H, W].
            num_inference_steps: Number of denoising steps.

        Returns:
            Image tensor [1, 3, H_img, W_img] in [0, 1] range.
        """
        device = self.device

        # 1. Encode prompt
        prompt_embeds = self.encode_prompt(prompt)  # list of 1 tensor

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

        # 3. Denoising loop (B=1, no CFG)
        for t in timesteps:
            timestep = (1000 - t.expand(1)) / 1000
            latent_input = latents.to(self.transformer.dtype).unsqueeze(
                2
            )  # [1, C, 1, H, W]

            noise_pred = self.transformer(
                [latent_input[0].to(device)],
                timestep.to(device),
                [prompt_embeds[0].to(device)],
                return_dict=False,
            )[0][
                0
            ]  # single output tensor

            noise_pred = -noise_pred.cpu().float().squeeze(1)  # [C, H, W]

            latents = self.scheduler.step(
                noise_pred.unsqueeze(0), t, latents, return_dict=False
            )[0]

        # 4. VAE decode (on CPU)
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae_scaling_factor) + self.vae_shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        return (image * 0.5 + 0.5).clamp(0, 1)
