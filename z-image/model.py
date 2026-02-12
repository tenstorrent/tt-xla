# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

# ---------------------------------------------------------------------------
# Scheduler (plain class, CPU only)
# ---------------------------------------------------------------------------


class Scheduler:
    """Wraps the diffusers scheduler with Z-Image's dynamic shift logic."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def compute_timesteps(self, latents, num_inference_steps):
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
        return self.scheduler.timesteps

    def step(self, noise_pred, t, latents):
        return self.scheduler.step(
            noise_pred.to(torch.float32), t, latents, return_dict=False
        )[0]


# ---------------------------------------------------------------------------
# Tokenizer (plain class, CPU only)
# ---------------------------------------------------------------------------


class Tokenizer:
    """Wraps the HuggingFace tokenizer with Z-Image's chat template."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, prompt):
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
        return tokens.input_ids, tokens.attention_mask.bool()


# ---------------------------------------------------------------------------
# TextEncoderModule (nn.Module)
# ---------------------------------------------------------------------------


class TextEncoderModule(nn.Module):
    """Wraps Qwen3 to run layers 0..N-2 directly, avoiding the
    hook-based output_hidden_states mechanism that breaks torch.compile.

    Returns the output of the second-to-last decoder layer, which is
    equivalent to text_encoder(..., output_hidden_states=True).hidden_states[-2].
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


# ---------------------------------------------------------------------------
# Patched attention processor (Step 1: direct SDPA, no dispatch_attention_fn)
# ---------------------------------------------------------------------------


class PatchedZSingleStreamAttnProcessor:
    """Drop-in replacement for ZSingleStreamAttnProcessor that uses
    F.scaled_dot_product_attention directly instead of dispatch_attention_fn."""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
    ):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE (original complex path)
        def apply_rotary_emb(x_in, freqs_cis):
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Direct SDPA (replaces dispatch_attention_fn)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output


def _patch_attention(transformer):
    """Replace all attention processors with direct-SDPA versions."""
    patched = PatchedZSingleStreamAttnProcessor()
    for block in (
        list(transformer.noise_refiner)
        + list(transformer.context_refiner)
        + list(transformer.layers)
    ):
        block.attention.set_processor(patched)


# ---------------------------------------------------------------------------
# TransformerModule (nn.Module)
# ---------------------------------------------------------------------------


class TransformerModule(nn.Module):
    """Thin wrapper around the original diffusers Z-Image transformer.

    Converts between the batched tensor interface used by ZImageModule
    and the List[Tensor] interface expected by the original model.
    """

    def __init__(self, pipe_transformer):
        super().__init__()
        self.transformer = pipe_transformer
        _patch_attention(self.transformer)

    def forward(self, x, t, cap_feats, cap_mask=None):
        """Forward pass with batched tensors.

        Args:
            x: [B, C, F, H, W] — batched latents.
            t: [B] — timestep values.
            cap_feats: [B, max_len, cap_feat_dim] — padded caption features.
            cap_mask: [B, max_len] — True for valid tokens. If None, inferred.

        Returns:
            [B, C, F, H, W] — denoised output.
        """
        if cap_mask is None:
            cap_mask = cap_feats.abs().sum(dim=-1) > 0

        # Convert batched latents to List[Tensor]
        x_list = list(x.unbind(dim=0))

        # Filter cap_feats per item using cap_mask (variable-length)
        cap_feats_list = [cap_feats[i][cap_mask[i]] for i in range(cap_feats.shape[0])]

        # Call original diffusers forward
        out_list = self.transformer(x_list, t, cap_feats_list, return_dict=False)[0]

        # Stack back to batched tensor [B, C, F, H, W]
        return torch.stack(out_list, dim=0)


# ---------------------------------------------------------------------------
# VAEDecoder (nn.Module)
# ---------------------------------------------------------------------------


class VAEDecoder(nn.Module):
    """Wraps the VAE decoder with scaling/shift and postprocessing."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.scaling_factor = vae.config.scaling_factor
        self.shift_factor = vae.config.shift_factor

    def forward(self, latents):
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.scaling_factor) + self.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)


# ---------------------------------------------------------------------------
# ZImageModule — orchestrator
# ---------------------------------------------------------------------------


class ZImageModule(nn.Module):
    """Orchestrates the full Z-Image pipeline.

    Holds all five submodules (tokenizer, text_encoder, scheduler,
    transformer, vae) and runs the complete text-to-image pipeline
    in forward().
    """

    def __init__(self, pipe, device):
        super().__init__()
        self.tokenizer = Tokenizer(pipe.tokenizer)
        self.text_encoder = TextEncoderModule(pipe.text_encoder)
        self.scheduler = Scheduler(pipe.scheduler)
        self.transformer = TransformerModule(pipe.transformer)
        self.vae = VAEDecoder(pipe.vae)
        self.device = device

    def encode_prompt(self, input_ids, attention_mask):
        """Run the text encoder and filter padding.

        Args:
            input_ids: Token IDs from Tokenizer.
            attention_mask: Boolean attention mask from Tokenizer.

        Returns:
            List of variable-length embeddings (one per batch item).
        """
        prompt_embeds = self.text_encoder(
            input_ids.to(self.device),
            attention_mask.to(self.device),
        )

        # Filter padding on CPU (variable-length output, can't be compiled)
        prompt_embeds = prompt_embeds.cpu()
        attention_mask = attention_mask.cpu()
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
        device = self.device

        # 1. Tokenize on CPU
        pos_ids, pos_mask = self.tokenizer(positive_prompt)
        neg_ids, neg_mask = self.tokenizer(negative_prompt)

        # 2. Encode on text encoder's device
        prompt_embeds = self.encode_prompt(pos_ids, pos_mask)
        negative_prompt_embeds = self.encode_prompt(neg_ids, neg_mask)

        # 3. Compute timesteps
        timesteps = self.scheduler.compute_timesteps(latents, num_inference_steps)
        do_cfg = guidance_scale > 1

        # 4. Denoising loop
        transformer_dtype = next(self.transformer.parameters()).dtype

        for t in timesteps:
            timestep = (1000 - t.expand(latents.shape[0])) / 1000

            if do_cfg:
                latent_input = (
                    latents.to(transformer_dtype).repeat(2, 1, 1, 1).unsqueeze(2)
                )
                cap_feats_list = prompt_embeds + negative_prompt_embeds
                timestep_input = timestep.repeat(2)
            else:
                latent_input = latents.to(transformer_dtype).unsqueeze(2)
                cap_feats_list = prompt_embeds
                timestep_input = timestep

            # Pad variable-length cap_feats to uniform length and build mask
            max_len = max(c.shape[0] for c in cap_feats_list)
            cap_feats_padded = torch.stack(
                [F.pad(c, (0, 0, 0, max_len - c.shape[0])) for c in cap_feats_list]
            )
            cap_mask = torch.stack(
                [
                    F.pad(
                        torch.ones(c.shape[0], dtype=torch.bool),
                        (0, max_len - c.shape[0]),
                    )
                    for c in cap_feats_list
                ]
            )

            # Run transformer (batched)
            noise_pred = self.transformer(
                latent_input.to(device),
                timestep_input.to(device),
                cap_feats_padded.to(device),
                cap_mask.to(device),
            )

            # CFG combine (back on CPU for scheduler)
            if do_cfg:
                bs = latents.shape[0]
                pos_out = noise_pred[:bs].cpu().float()
                neg_out = noise_pred[bs:].cpu().float()
                noise_pred = pos_out + guidance_scale * (pos_out - neg_out)
            else:
                noise_pred = noise_pred.cpu().float()

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred

            latents = self.scheduler.step(noise_pred, t, latents)

        # 5. VAE decode (CPU)
        return self.vae(latents)
