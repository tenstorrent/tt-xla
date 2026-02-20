# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


class RealRopeEmbedder(nn.Module):
    """Real-valued replacement for diffusers' RopeEmbedder.

    Precomputes cos/sin lookup tables in __init__ and registers them as buffers
    so they move with .to(device) automatically. No complex64, no lazy init.
    """

    def __init__(self, theta: float, axes_dims: List[int], axes_lens: List[int]):
        super().__init__()
        self.axes_dims = axes_dims
        for i, (d, e) in enumerate(zip(axes_dims, axes_lens)):
            freqs = 1.0 / (
                theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
            )
            timestep = torch.arange(e, device="cpu", dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()  # [e, d/2]
            self.register_buffer(f"cos_{i}", freqs.cos())  # [e, d/2]
            self.register_buffer(f"sin_{i}", freqs.sin())  # [e, d/2]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [seq, num_axes]
        # Gather per-axis, then concatenate as [all_cos, all_sin] so that
        # _apply_rotary_emb can split in half to get cos and sin.
        cos_parts = []
        sin_parts = []
        for i in range(len(self.axes_dims)):
            cos_parts.append(getattr(self, f"cos_{i}")[ids[:, i]])
            sin_parts.append(getattr(self, f"sin_{i}")[ids[:, i]])
        return torch.cat(cos_parts + sin_parts, dim=-1)


class RealRopeAttnProcessor:
    """Real-valued replacement for ZSingleStreamAttnProcessor.

    Identical logic except apply_rotary_emb uses real cos/sin arithmetic
    instead of complex64 multiply.

    Not an nn.Module: Attention.forward introspects __call__ signature to
    discover which kwargs (like freqs_cis) to forward from cross_attention_kwargs.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        if freqs_cis is not None:
            query = self._apply_rotary_emb(query, freqs_cis)
            key = self._apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

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
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output

    @staticmethod
    def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # x_in: [batch, seq, heads, head_dim]
        # freqs_cis: [batch, seq, head_dim] — first half cos, second half sin
        half = freqs_cis.shape[-1] // 2
        cos = freqs_cis[..., :half].unsqueeze(2)  # [batch, seq, 1, half]
        sin = freqs_cis[..., half:].unsqueeze(2)

        x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
        x_real = x[..., 0]  # [batch, seq, heads, half]
        x_imag = x[..., 1]

        # (a+bi)(cos+i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos
        out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
        return out.type_as(x_in)


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


SEQ_MULTI_OF = 32


def _make_transformer_forward(transformer, cap_ori_len, image_shape):
    """Build a graph-break-free replacement for ZImageTransformer2DModel.forward.

    The original forward calls patchify_and_embed which uses len(), Python ifs,
    and variable-length list ops — all of which cause XLA graph breaks.
    Since we always run B=1 with fixed shapes, we precompute everything here
    and return a closure that does straight-line tensor math only.
    """
    C, F, H, W = image_shape  # e.g. 16, 1, 160, 90
    patch_size = 2
    f_patch_size = 1
    pH = pW = patch_size
    pF = f_patch_size

    # Caption padding
    cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
    cap_total = cap_ori_len + cap_padding_len

    # Image token dimensions
    F_tokens = F // pF
    H_tokens = H // pH
    W_tokens = W // pW
    image_ori_len = F_tokens * H_tokens * W_tokens
    image_padding_len = (-image_ori_len) % SEQ_MULTI_OF
    image_total = image_ori_len + image_padding_len

    # Precompute position IDs (int32 tensors, will be registered as buffers)
    # Image pos IDs: coordinate grid [F_tokens, H_tokens, W_tokens, 3]
    # with start = (cap_total + 1, 0, 0), flattened to [image_ori_len, 3]
    # then padded with zeros to [image_total, 3]
    image_ori_pos_ids = torch.stack(
        torch.meshgrid(
            torch.arange(cap_total + 1, cap_total + 1 + F_tokens, dtype=torch.int32),
            torch.arange(0, H_tokens, dtype=torch.int32),
            torch.arange(0, W_tokens, dtype=torch.int32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(image_ori_len, 3)
    zero_pad = torch.zeros(image_padding_len, 3, dtype=torch.int32)
    image_pos_ids = torch.cat([image_ori_pos_ids, zero_pad], dim=0)

    # Caption pos IDs: [cap_total, 3] with start=(1, 0, 0)
    cap_pos_ids = torch.stack(
        torch.meshgrid(
            torch.arange(1, 1 + cap_total, dtype=torch.int32),
            torch.arange(0, 1, dtype=torch.int32),
            torch.arange(0, 1, dtype=torch.int32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(cap_total, 3)

    # Pad masks (bool)
    image_pad_mask = torch.cat(
        [
            torch.zeros(image_ori_len, dtype=torch.bool),
            torch.ones(image_padding_len, dtype=torch.bool),
        ]
    )
    cap_pad_mask = torch.cat(
        [
            torch.zeros(cap_ori_len, dtype=torch.bool),
            torch.ones(cap_padding_len, dtype=torch.bool),
        ]
    )

    # Attention masks [1, seq_len] (True = attend)
    # The original forward sets mask=1 for the full padded length (x_item_seqlens
    # includes padding), so all positions attend. Pad tokens are handled by
    # replacing their embeddings with x_pad_token/cap_pad_token, not by masking.
    x_attn_mask = torch.ones(1, image_total, dtype=torch.bool)
    cap_attn_mask = torch.ones(1, cap_total, dtype=torch.bool)

    unified_seq_len = image_total + cap_total
    unified_attn_mask = torch.cat([x_attn_mask, cap_attn_mask], dim=1)

    # Store references to transformer submodules
    x_embedder = transformer.all_x_embedder[f"{patch_size}-{f_patch_size}"]
    final_layer = transformer.all_final_layer[f"{patch_size}-{f_patch_size}"]
    cap_embedder = transformer.cap_embedder
    x_pad_token = transformer.x_pad_token
    cap_pad_token = transformer.cap_pad_token
    rope_embedder = transformer.rope_embedder
    t_embedder = transformer.t_embedder
    t_scale = transformer.t_scale
    noise_refiner = transformer.noise_refiner
    context_refiner = transformer.context_refiner
    layers = transformer.layers
    out_channels = transformer.out_channels

    # Register precomputed tensors as buffers on the transformer
    transformer.register_buffer("_image_pos_ids", image_pos_ids)
    transformer.register_buffer("_cap_pos_ids", cap_pos_ids)
    transformer.register_buffer("_image_pad_mask", image_pad_mask)
    transformer.register_buffer("_cap_pad_mask", cap_pad_mask)
    transformer.register_buffer("_x_attn_mask", x_attn_mask)
    transformer.register_buffer("_cap_attn_mask", cap_attn_mask)
    transformer.register_buffer("_unified_attn_mask", unified_attn_mask)

    def forward(self, x, t, cap_feats, patch_size=2, f_patch_size=1, return_dict=True):
        image = x[0]  # [C, F, H, W]
        cap_feat = cap_feats[0]  # [cap_ori_len, 2560]
        device = image.device

        # 1. Timestep embedding
        t_emb = t_embedder(t * t_scale)

        # 2. Patchify image: [C, F, H, W] -> [image_ori_len, patch_dim]
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
            image_ori_len, pF * pH * pW * C
        )

        # 3. Pad image features
        image = torch.cat([image, image[-1:].expand(image_padding_len, -1)], dim=0)

        # 4. Pad caption features
        cap_feat = torch.cat(
            [cap_feat, cap_feat[-1:].expand(cap_padding_len, -1)], dim=0
        )

        # 5. Embed
        x = x_embedder(image)
        adaln_input = t_emb.type_as(x)
        x[self._image_pad_mask] = x_pad_token
        cap = cap_embedder(cap_feat)
        cap[self._cap_pad_mask] = cap_pad_token

        # 6. RoPE
        x_freqs_cis = rope_embedder(self._image_pos_ids)
        cap_freqs_cis = rope_embedder(self._cap_pos_ids)

        # 7. Add batch dim: [seq, dim] -> [1, seq, dim]
        x = x.unsqueeze(0)
        x_freqs_cis = x_freqs_cis.unsqueeze(0)
        cap = cap.unsqueeze(0)
        cap_freqs_cis = cap_freqs_cis.unsqueeze(0)

        # 8. Noise refiner
        for layer in noise_refiner:
            x = layer(x, self._x_attn_mask, x_freqs_cis, adaln_input)

        # 9. Context refiner
        for layer in context_refiner:
            cap = layer(cap, self._cap_attn_mask, cap_freqs_cis)

        # 10. Unified layers
        unified = torch.cat([x, cap], dim=1)
        unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)

        for layer in layers:
            unified = layer(
                unified, self._unified_attn_mask, unified_freqs_cis, adaln_input
            )

        # 11. Final layer + unpatchify
        unified = final_layer(unified, adaln_input)
        # Take only image tokens, remove padding
        out = unified[0, :image_ori_len]  # [image_ori_len, patch_dim]
        out = (
            out.view(F_tokens, H_tokens, W_tokens, pF, pH, pW, out_channels)
            .permute(6, 0, 3, 1, 4, 2, 5)
            .reshape(out_channels, F, H, W)
        )

        if not return_dict:
            return ([out],)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=[out])

    return forward


class ZImageModule(nn.Module):
    """Wraps the full Z-Image pipeline into a single nn.Module.

    __init__ extracts all components from a ZImagePipeline and applies
    the RoPE monkey-patch (complex64 -> real-valued) for XLA compatibility,
    and replaces the transformer's forward with a graph-break-free version.

    forward() runs the complete pipeline: text encoding, denoising loop with CFG,
    VAE decode, and postprocessing to a raw image tensor. The transformer inputs
    are automatically moved to/from whatever device it lives on.
    """

    def __init__(self, pipe, device, cap_len=None, image_shape=None):
        super().__init__()
        self.text_encoder_module = TextEncoderModule(pipe.text_encoder)
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
        self.vae_scaling_factor = pipe.vae.config.scaling_factor
        self.vae_shift_factor = pipe.vae.config.shift_factor

        self.device = device

        # Replace complex64 RoPE with real-valued equivalent for XLA compatibility
        cfg = self.transformer.config
        self.transformer.rope_embedder = RealRopeEmbedder(
            theta=cfg.rope_theta,
            axes_dims=cfg.axes_dims,
            axes_lens=cfg.axes_lens,
        )
        for block_list in [
            self.transformer.noise_refiner,
            self.transformer.context_refiner,
            self.transformer.layers,
        ]:
            for block in block_list:
                block.attention.set_processor(RealRopeAttnProcessor())

        # Replace transformer forward with graph-break-free version
        if cap_len is not None and image_shape is not None:
            import types

            new_forward = _make_transformer_forward(
                self.transformer,
                cap_len,
                image_shape,
            )
            self.transformer.forward = types.MethodType(new_forward, self.transformer)

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

    def _apply_transformer_forward_replacement(self, cap_len, image_shape):
        """Apply graph-break-free transformer forward if not already done."""
        if not hasattr(self.transformer, "_image_pos_ids"):
            import types

            new_fwd = _make_transformer_forward(
                self.transformer,
                cap_len,
                image_shape,
            )
            self.transformer.forward = types.MethodType(new_fwd, self.transformer)

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

        # Apply graph-break-free forward now that we know the shapes
        C = latents.shape[1]
        H, W = latents.shape[2], latents.shape[3]
        self._apply_transformer_forward_replacement(
            cap_len=prompt_embeds[0].shape[0],
            image_shape=(C, 1, H, W),
        )

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
