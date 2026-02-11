# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from diffusers.models.transformers import transformer_z_image
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift


def patch_rope_real_valued():
    """Monkey-patch Z-Image's RoPE to use real-valued cos/sin arithmetic.

    The original implementation uses torch.complex64 (via torch.polar,
    view_as_complex, view_as_real) which XLA/PJRT doesn't support.
    This replaces it with mathematically equivalent real-valued operations.

    Each axis stores cos and sin separately as a (cos, sin) tuple of real
    tensors with shape [end, d//2]. RopeEmbedder.__call__ is patched to
    index-gather each axis and concatenate as [all_cos | all_sin] so that
    apply_rotary_emb_real can split at the midpoint.
    """

    # 1. Patch precompute_freqs_cis: store (cos, sin) tuples per axis
    @staticmethod
    def precompute_freqs_cis_real(dim, end, theta=256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (
                    theta
                    ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
                )
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                # Use torch.polar to match the original implementation exactly,
                # then extract real (cos) and imag (sin) parts.
                polar = torch.polar(torch.ones_like(freqs), freqs)
                freqs_cis.append((polar.real, polar.imag))
            return freqs_cis

    # 2. Patch RopeEmbedder.__call__ to produce [all_cos | all_sin] layout
    def rope_call_real(self, ids):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(
                self.axes_dims, self.axes_lens, theta=self.theta
            )
            self.freqs_cis = [(c.to(device), s.to(device)) for c, s in self.freqs_cis]
        else:
            if self.freqs_cis[0][0].device != device:
                self.freqs_cis = [
                    (c.to(device), s.to(device)) for c, s in self.freqs_cis
                ]

        cos_parts = []
        sin_parts = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            cos_i, sin_i = self.freqs_cis[i]
            cos_parts.append(cos_i[index])
            sin_parts.append(sin_i[index])
        return torch.cat(cos_parts + sin_parts, dim=-1)

    # 3. Patch the attention processor to use real-valued rotary embedding
    def attn_call_real(
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

        # Real-valued RoPE: freqs_cis has [all_cos | all_sin] in last dim
        def apply_rotary_emb_real(x_in, freqs_cis):
            half = freqs_cis.shape[-1] // 2
            cos = freqs_cis[..., :half].unsqueeze(2)
            sin = freqs_cis[..., half:].unsqueeze(2)

            x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
            x_real = x[..., 0]
            x_imag = x[..., 1]

            out_real = x_real * cos - x_imag * sin
            out_imag = x_real * sin + x_imag * cos
            out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
            return out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb_real(query, freqs_cis)
            key = apply_rotary_emb_real(key, freqs_cis)

        dt = query.dtype
        query, key = query.to(dt), key.to(dt)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        from diffusers.models.attention_dispatch import dispatch_attention_fn

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dt)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output

    # Apply patches
    transformer_z_image.RopeEmbedder.precompute_freqs_cis = precompute_freqs_cis_real
    transformer_z_image.RopeEmbedder.__call__ = rope_call_real
    transformer_z_image.ZSingleStreamAttnProcessor.__call__ = attn_call_real
    print("Patched Z-Image RoPE to use real-valued arithmetic (no complex64)")


class ZImageModule(nn.Module):
    """Wraps the full Z-Image pipeline into a single nn.Module.

    __init__ extracts all components from a ZImagePipeline and applies
    the RoPE monkey-patch (complex64 -> real-valued) for XLA compatibility.

    forward() runs the complete pipeline: text encoding, denoising loop with CFG,
    VAE decode, and postprocessing to a raw image tensor. The transformer inputs
    are automatically moved to/from whatever device it lives on.
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

        # Patch RoPE to avoid complex64 (not supported by XLA/PJRT)
        patch_rope_real_valued()
        # Clear any cached complex64 freqs_cis so the patched version recomputes
        self.transformer.rope_embedder.freqs_cis = None

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

    @property
    def device(self):
        """Device where the transformer lives (CPU or TT)."""
        return next(self.transformer.parameters()).device

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

        # 1. Encode prompts (always on CPU — tokenizer + hidden_states not traceable)
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
            latent_list = [x.to(device) for x in latent_model_input.unbind(dim=0)]
            prompt_embeds_device = [x.to(device) for x in prompt_embeds_input]
            timestep_device = timestep_input.to(device)

            # Transformer forward (on whatever device it lives on)
            model_out_list = self.transformer(
                latent_list, timestep_device, prompt_embeds_device, return_dict=False
            )[0]

            if do_cfg:
                # CFG combine (back on CPU for scheduler)
                actual_batch_size = latents.shape[0]
                pos_out = model_out_list[:actual_batch_size]
                neg_out = model_out_list[actual_batch_size:]

                noise_pred = []
                for j in range(actual_batch_size):
                    pos = pos_out[j].cpu().float()
                    neg = neg_out[j].cpu().float()
                    noise_pred.append(pos + guidance_scale * (pos - neg))
                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack(
                    [x.cpu().float() for x in model_out_list], dim=0
                )

            # Squeeze temporal dim + negate for flow matching
            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred

            # Scheduler step (on CPU — uses numpy internally)
            latents = self.scheduler.step(
                noise_pred.to(torch.float32), t, latents, return_dict=False
            )[0]

        # 4. VAE decode (on CPU)
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae_scaling_factor) + self.vae_shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        # 5. Postprocess: denormalize to [0, 1]
        image = (image * 0.5 + 0.5).clamp(0, 1)

        return image
