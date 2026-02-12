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

        # Apply RoPE (real-valued storage: freqs_cis is [N, rope_dim] with [all_cos | all_sin])
        def apply_rotary_emb(x_in, freqs_cis):
            half = freqs_cis.shape[-1] // 2
            cos = freqs_cis[..., :half].unsqueeze(2)
            sin = freqs_cis[..., half:].unsqueeze(2)
            x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
            x_real, x_imag = x[..., 0], x[..., 1]
            out = torch.stack(
                [x_real * cos - x_imag * sin, x_real * sin + x_imag * cos],
                dim=-1,
            ).flatten(3)
            return out.type_as(x_in)

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
# Step 3: Real-valued RoPE storage (no complex64)
# ---------------------------------------------------------------------------


class RealRopeEmbedder:
    """RoPE embedder using (cos, sin) tuples instead of complex64."""

    def __init__(self, theta, axes_dims, axes_lens):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (
                    theta
                    ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
                )
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                polar = torch.polar(torch.ones_like(freqs), freqs)
                freqs_cis.append((polar.real, polar.imag))
            return freqs_cis

    def __call__(self, ids):
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


def _patch_rope(transformer):
    """Replace rope_embedder with real-valued version (no complex64)."""
    old = transformer.rope_embedder
    transformer.rope_embedder = RealRopeEmbedder(
        theta=old.theta, axes_dims=old.axes_dims, axes_lens=old.axes_lens
    )


# ---------------------------------------------------------------------------
# TransformerModule (nn.Module)
# ---------------------------------------------------------------------------


SEQ_MULTI_OF = 32


class TransformerModule(nn.Module):
    """Wrapper around the original diffusers Z-Image transformer.

    Provides a batched tensor interface and calls the original model's
    submodules directly, replicating the original forward logic exactly
    but using batched tensors instead of List[Tensor].
    """

    def __init__(self, pipe_transformer):
        super().__init__()
        self.transformer = pipe_transformer
        _patch_attention(self.transformer)
        _patch_rope(self.transformer)

    def forward(self, x, t, cap_feats, cap_mask=None):
        """Batched forward pass.

        Args:
            x: [B, C, F, H, W] — batched latents (all same spatial size).
            t: [B] — timestep values.
            cap_feats: [B, max_len, cap_feat_dim] — padded caption features.
            cap_mask: [B, max_len] — True for valid tokens. If None, inferred.

        Returns:
            [B, C, F, H, W] — denoised output.
        """
        if cap_mask is None:
            cap_mask = cap_feats.abs().sum(dim=-1) > 0

        mdl = self.transformer
        patch_size = mdl.all_patch_size[0]
        f_patch_size = mdl.all_f_patch_size[0]
        pH = pW = patch_size
        pF = f_patch_size
        B, C, Fr, H, W = x.shape
        device = x.device

        # --- timestep embed ---
        t_scaled = t * mdl.t_scale
        t_emb = mdl.t_embedder(t_scaled)

        # === patchify_and_embed (batched, mirrors original per-item logic) ===
        F_tokens = Fr // pF
        H_tokens = H // pH
        W_tokens = W // pW
        image_ori_len = F_tokens * H_tokens * W_tokens
        image_padding_len = (-image_ori_len) % SEQ_MULTI_OF
        image_padded_len = image_ori_len + image_padding_len

        # Patchify: [B, C, F, H, W] -> [B, F_t*H_t*W_t, pF*pH*pW*C]
        x_patched = x.view(B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        x_patched = x_patched.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
            B, image_ori_len, pF * pH * pW * C
        )

        # Image alignment padding (repeat last token)
        if image_padding_len > 0:
            pad_tokens = x_patched[:, -1:, :].expand(B, image_padding_len, -1)
            x_patched = torch.cat([x_patched, pad_tokens], dim=1)

        # Image inner pad mask
        x_inner_pad_mask = torch.zeros(
            (B, image_padded_len), dtype=torch.bool, device=device
        )
        if image_padding_len > 0:
            x_inner_pad_mask[:, image_ori_len:] = True

        # Per-item caption processing (variable lengths require per-item logic)
        cap_valid_lens = cap_mask.sum(dim=1)  # [B]
        cap_padding_lens = (-cap_valid_lens) % SEQ_MULTI_OF  # [B]
        cap_aligned_lens = cap_valid_lens + cap_padding_lens  # [B]

        # Build per-item caption outputs matching original exactly
        all_cap_feats_out = []
        all_cap_pos_ids = []
        all_cap_inner_pad_mask = []
        cap_item_seqlens = []

        for i in range(B):
            valid = cap_valid_lens[i].item()
            pad_len = cap_padding_lens[i].item()
            aligned = valid + pad_len

            # Padded feature: valid tokens + repeat last valid token
            cap_feat_i = cap_feats[i, :valid]  # [valid, dim]
            if pad_len > 0:
                cap_padded_i = torch.cat(
                    [cap_feat_i, cap_feat_i[-1:].expand(pad_len, -1)], dim=0
                )
            else:
                cap_padded_i = cap_feat_i
            all_cap_feats_out.append(cap_padded_i)
            cap_item_seqlens.append(aligned)

            # Position IDs: (1..aligned, 0, 0)
            cap_pos_i = mdl.create_coordinate_grid(
                size=(aligned, 1, 1), start=(1, 0, 0), device=device
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_pos_i)

            # Inner pad mask: True for alignment padding
            mask_i = torch.zeros(aligned, dtype=torch.bool, device=device)
            if pad_len > 0:
                mask_i[valid:] = True
            all_cap_inner_pad_mask.append(mask_i)

        # Image position IDs (per-item f-axis offset = aligned_cap_len + 1)
        base_image_grid = mdl.create_coordinate_grid(
            size=(F_tokens, H_tokens, W_tokens), start=(0, 0, 0), device=device
        ).flatten(
            0, 2
        )  # [image_ori_len, 3]

        if image_padding_len > 0:
            zero_pos = torch.zeros(
                (image_padding_len, 3), dtype=torch.int32, device=device
            )
            base_grid_padded = torch.cat([base_image_grid, zero_pos], dim=0)
        else:
            base_grid_padded = base_image_grid

        all_image_pos_ids = []
        for i in range(B):
            pos_i = base_grid_padded.clone()
            f_offset = cap_aligned_lens[i].item() + 1
            pos_i[:image_ori_len, 0] += f_offset
            all_image_pos_ids.append(pos_i)

        # === x embed & refine (matches original: cat -> embed -> pad_token -> split -> pad_sequence) ===
        x_seqlen = image_padded_len  # same for all items since spatial size is uniform
        x_flat = x_patched.reshape(B * x_seqlen, -1)  # cat equivalent
        x_embedded = mdl.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_flat)
        adaln_input = t_emb.type_as(x_embedded)

        # Replace alignment padding with x_pad_token (in-place on flat tensor, matching original)
        flat_pad_mask = x_inner_pad_mask.reshape(-1)
        x_embedded[flat_pad_mask] = mdl.x_pad_token

        # Reshape back to [B, x_seqlen, dim] (equivalent to split + pad_sequence since all same length)
        x_tokens = x_embedded.reshape(B, x_seqlen, -1)

        # RoPE for image
        x_pos_ids_cat = torch.cat(all_image_pos_ids, dim=0)  # [B * x_seqlen, 3]
        x_freqs_cis = mdl.rope_embedder(x_pos_ids_cat).reshape(B, x_seqlen, -1)

        # Attention mask: all True (all items same length, no batch-level padding)
        x_attn_mask = torch.ones((B, x_seqlen), dtype=torch.bool, device=device)

        for layer in mdl.noise_refiner:
            x_tokens = layer(x_tokens, x_attn_mask, x_freqs_cis, adaln_input)

        # === cap embed & refine (cat -> embed -> pad_token -> split -> pad_sequence) ===
        cap_max_seqlen = max(cap_item_seqlens)

        cap_flat = torch.cat(all_cap_feats_out, dim=0)  # [sum_cap_lens, cap_feat_dim]
        cap_embedded = mdl.cap_embedder(cap_flat)

        # Replace alignment padding with cap_pad_token
        cap_flat_pad_mask = torch.cat(all_cap_inner_pad_mask, dim=0)
        cap_embedded[cap_flat_pad_mask] = mdl.cap_pad_token

        # Split back to per-item and pad_sequence to uniform length
        cap_splits = list(cap_embedded.split(cap_item_seqlens, dim=0))
        cap_tokens = torch.nn.utils.rnn.pad_sequence(
            cap_splits, batch_first=True, padding_value=0.0
        )

        # RoPE for captions
        cap_pos_ids_cat = torch.cat(all_cap_pos_ids, dim=0)
        cap_freqs_splits = list(
            mdl.rope_embedder(cap_pos_ids_cat).split(cap_item_seqlens, dim=0)
        )
        cap_freqs_cis = torch.nn.utils.rnn.pad_sequence(
            cap_freqs_splits, batch_first=True, padding_value=0.0
        )
        cap_freqs_cis = cap_freqs_cis[:, : cap_tokens.shape[1]]

        # Caption attention mask
        cap_attn_mask = torch.zeros(
            (B, cap_max_seqlen), dtype=torch.bool, device=device
        )
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = True

        for layer in mdl.context_refiner:
            cap_tokens = layer(cap_tokens, cap_attn_mask, cap_freqs_cis)

        # === unified assembly (matches original exactly) ===
        unified_item_seqlens = [x_seqlen + cap_len for cap_len in cap_item_seqlens]
        unified_max_len = max(unified_item_seqlens)

        unified = []
        unified_freqs = []
        for i in range(B):
            x_len = x_seqlen  # same for all items
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x_tokens[i, :x_len], cap_tokens[i, :cap_len]]))
            unified_freqs.append(
                torch.cat([x_freqs_cis[i, :x_len], cap_freqs_cis[i, :cap_len]])
            )

        unified = torch.nn.utils.rnn.pad_sequence(
            unified, batch_first=True, padding_value=0.0
        )
        unified_freqs_cis = torch.nn.utils.rnn.pad_sequence(
            unified_freqs, batch_first=True, padding_value=0.0
        )
        unified_attn_mask = torch.zeros(
            (B, unified_max_len), dtype=torch.bool, device=device
        )
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = True

        # === main transformer layers ===
        for layer in mdl.layers:
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        # === final layer + unpatchify ===
        unified = mdl.all_final_layer[f"{patch_size}-{f_patch_size}"](
            unified, adaln_input
        )

        # Extract image tokens and unpatchify
        # Image tokens are at the start of each item (x_seqlen tokens)
        x_out = unified[:, :x_seqlen, :]

        # Batched unpatchify: [B, x_seqlen, patch_channels] -> [B, C, F, H, W]
        x_out = x_out[:, :image_ori_len]
        x_out = x_out.view(
            B, F_tokens, H_tokens, W_tokens, pF, pH, pW, mdl.out_channels
        )
        x_out = x_out.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            B, mdl.out_channels, Fr, H, W
        )

        return x_out


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
