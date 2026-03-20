# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Wan video generation pipeline with TT device acceleration.

NN modules (transformer, optionally text_encoder and VAE) are compiled for TT
and run on xla_device. All other logic (scheduler, CFG blending, latent prep,
timestep expansion) stays on CPU.

Mirrors the CPU-only wan_t2v_pipeline.py with tt_cast / cpu_cast wrappers
around NN forward calls, following the same pattern as sd_v1_4_pipeline.py.
"""

import html
import re
from pathlib import Path
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.transformers.transformer_wan import (
    WanAttnProcessor,
    _get_qkv_projections,
    _get_added_kv_projections,
)
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel


def basic_clean(text):
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prompt_clean(text):
    return whitespace_clean(basic_clean(text))


class TTWanAttnProcessor:
    """TT-compatible WanAttnProcessor that avoids in-place mutations.

    The stock WanAttnProcessor uses torch.empty_like + strided slice assignment
    in apply_rotary_emb, which causes 'double free or corruption' on the TT XLA
    backend.  This version replaces that with purely functional torch.stack +
    flatten.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out_even = x1 * cos - x2 * sin
                out_odd = x1 * sin + x2 * cos
                return torch.stack([out_even, out_odd], dim=-1).flatten(-2).type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query, key_img, value_img,
                attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend,
                parallel_config=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _patch_transformer_for_tt(transformer):
    """Replace WanAttnProcessor with TTWanAttnProcessor on all attention layers."""
    patched = TTWanAttnProcessor()
    for block in transformer.blocks:
        block.attn1.processor = patched
        block.attn2.processor = patched


def _patch_vae_for_tt(vae):
    """Patch VAE _decode to process all frames at once, avoiding CACHE_T issues.

    The stock _decode processes latent frames one-at-a-time in a loop, passing
    feat_cache to the decoder.  Inside the decoder, cache management uses
    ``x[:, :, -CACHE_T:, :, :]`` (CACHE_T=2) on single-frame tensors (temporal
    dim=1).  The TT XLA backend rejects the out-of-bounds negative start index.

    Fix: bypass the frame-by-frame loop and pass all frames at once with
    feat_cache=None.  The decoder's causal convolutions apply full left-padding
    instead, which is mathematically equivalent and fully functional on TT.
    """
    import types

    def _decode_tt(self, z, return_dict=True):
        _, _, num_frame, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        x = self.post_quant_conv(z)
        out = self.decoder(x)

        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)

        out = torch.clamp(out, min=-1.0, max=1.0)

        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)

    vae._decode = types.MethodType(_decode_tt, vae)


class WanTTConfig:
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        device: str = "cpu",
        vae_on_tt: bool = False,
        text_encoder_on_tt: bool = False,
        expand_timesteps: bool = True,
    ):
        self.model_id = model_id
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.device = device
        self.vae_on_tt = vae_on_tt
        self.text_encoder_on_tt = text_encoder_on_tt
        self.expand_timesteps = expand_timesteps


class WanT2VTTPipeline:
    """Pipeline for video generation with Wan, running NN modules on TT device.

    Transformer is always compiled and placed on TT.
    Text encoder and VAE placement is controlled via config flags.
    Scheduler, CFG blending, latent preparation, and timestep expansion
    remain on CPU.
    """

    def __init__(self, config: WanTTConfig):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id
        self.vae_on_tt = config.vae_on_tt
        self.text_encoder_on_tt = config.text_encoder_on_tt
        self.expand_timesteps = config.expand_timesteps

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizer()
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

    def load_models(self):
        # --- Text encoder ---
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16 if self.text_encoder_on_tt else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        if self.text_encoder_on_tt:
            self.text_encoder.compile(backend="tt")
            self.text_encoder = self.text_encoder.to(xm.xla_device())

        # --- Transformer (always on TT) ---
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        _patch_transformer_for_tt(self.transformer)
        self.transformer.compile(backend="tt")
        self.transformer = self.transformer.to(xm.xla_device())

        # --- VAE ---
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        if self.vae_on_tt:
            _patch_vae_for_tt(self.vae)
            self.vae.compile(backend="tt")
            self.vae = self.vae.to(xm.xla_device())

    def load_scheduler(self):
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------
    def _encode_prompt_single(self, prompt_list, max_sequence_length):
        """Tokenize + T5-encode a list of strings, trim & re-pad to zeros."""
        prompt_list = [prompt_clean(u) for u in prompt_list]

        text_inputs = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        mask = text_inputs.attention_mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        if self.text_encoder_on_tt:
            text_input_ids = text_input_ids.to(xm.xla_device())
            mask = mask.to(xm.xla_device())

        embeds = self.text_encoder(text_input_ids, mask).last_hidden_state

        if self.text_encoder_on_tt:
            embeds = embeds.to("cpu").to(dtype=torch.float32)
        else:
            embeds = embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        # Trim to actual sequence length and re-pad with zeros (CPU)
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        embeds = torch.stack(
            [
                torch.cat(
                    [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]
                )
                for u in embeds
            ],
            dim=0,
        )
        return embeds

    def encode_prompt(self, prompt, negative_prompt="", max_sequence_length=512):
        """Encode prompt + negative prompt."""
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt_single(prompt_list, max_sequence_length)

        neg_list = (
            [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        )
        if not neg_list or neg_list[0] is None:
            neg_list = [""] * len(prompt_list)
        negative_prompt_embeds = self._encode_prompt_single(
            neg_list, max_sequence_length
        )

        return prompt_embeds, negative_prompt_embeds

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        max_sequence_length: int = 512,
        output_type: str = "np",
    ):
        """Generate video from a text prompt with TT-accelerated NN modules."""
        height = height or self.config.height
        width = width or self.config.width
        num_frames = num_frames or self.config.num_frames
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_cfg = guidance_scale > 1.0

        tt_cast = lambda x: (
            x.to(dtype=torch.bfloat16).to(device=xm.xla_device())
            if x.device == torch.device("cpu")
            else x.to(dtype=torch.bfloat16)
        )
        cpu_cast = lambda x: x.to("cpu").to(dtype=torch.float32)

        with torch.no_grad():
            # --- 1. Adjust num_frames ---
            if num_frames % self.vae_scale_factor_temporal != 1:
                num_frames = (
                    num_frames
                    // self.vae_scale_factor_temporal
                    * self.vae_scale_factor_temporal
                    + 1
                )
            num_frames = max(num_frames, 1)

            # --- 2. Align height/width to patch-size multiples ---
            patch_size = self.transformer.config.patch_size
            h_multiple = self.vae_scale_factor_spatial * patch_size[1]
            w_multiple = self.vae_scale_factor_spatial * patch_size[2]
            height = height // h_multiple * h_multiple
            width = width // w_multiple * w_multiple

            # --- 3. Generator (CPU) ---
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # --- 4. Text encoding (T5, may run on TT) ---
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt, negative_prompt, max_sequence_length
            )

            # --- 5. Prepare timesteps (CPU) ---
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            # --- 6. Prepare latents (CPU, float32) ---
            num_channels_latents = self.transformer.config.in_channels
            num_latent_frames = (
                (num_frames - 1) // self.vae_scale_factor_temporal + 1
            )
            latent_shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )
            latents = torch.randn(
                latent_shape,
                generator=generator,
                dtype=torch.float32,
                device=self.device,
            )

            # mask for expand_timesteps (CPU)
            mask = torch.ones_like(latents)

            # --- 7. Denoising loop ---
            for i, t in enumerate(timesteps):
                # Timestep expansion (CPU)
                if self.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)
                else:
                    timestep = t.expand(batch_size)

                # CPU → TT
                latent_model_input = tt_cast(latents)
                timestep_tt = tt_cast(timestep)
                prompt_embeds_tt = tt_cast(prompt_embeds)

                # Conditional pass (TT)
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_tt,
                        encoder_hidden_states=prompt_embeds_tt,
                        return_dict=False,
                    )[0]

                # TT → CPU
                noise_pred = cpu_cast(noise_pred)

                # Unconditional pass for CFG
                if do_cfg:
                    neg_embeds_tt = tt_cast(negative_prompt_embeds)

                    with self.transformer.cache_context("uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep_tt,
                            encoder_hidden_states=neg_embeds_tt,
                            return_dict=False,
                        )[0]

                    # TT → CPU
                    noise_uncond = cpu_cast(noise_uncond)

                    # CFG blending (CPU)
                    noise_pred = noise_uncond + guidance_scale * (
                        noise_pred - noise_uncond
                    )

                # Scheduler step (CPU, float32)
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                print(f"  Step {i + 1}/{num_inference_steps}")

            # --- 8. VAE decode ---
            if output_type != "latent":
                latents_vae = latents.to(dtype=torch.float32)
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents_vae.device, latents_vae.dtype)
                )
                latents_std = (
                    1.0
                    / torch.tensor(self.vae.config.latents_std)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents_vae.device, latents_vae.dtype)
                )
                latents_vae = latents_vae / latents_std + latents_mean

                if self.vae_on_tt:
                    latents_vae = latents_vae.to(device=xm.xla_device())

                video = self.vae.decode(latents_vae, return_dict=False)[0]

                if self.vae_on_tt:
                    video = cpu_cast(video)

                video = self.video_processor.postprocess_video(
                    video, output_type=output_type
                )
            else:
                video = latents

            return video


# ======================================================================
# Convenience helpers
# ======================================================================


def run_wan_tt_pipeline(
    output_path: str = "wan_tt_output.mp4",
    num_inference_steps: int = 50,
    output_type: str = "np",
):
    """Run Wan pipeline with TT acceleration and optionally save the video."""
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    config = WanTTConfig(device="cpu")
    pipeline = WanT2VTTPipeline(config=config)
    pipeline.setup()

    video = pipeline.generate(
        prompt="A cat sitting on a sunny windowsill",
        negative_prompt="",
        guidance_scale=5.0,
        num_inference_steps=num_inference_steps,
        seed=42,
        output_type=output_type,
    )

    if output_type != "latent":
        export_to_video(video, output_path, fps=16)
        print(f"Video saved to {output_path}")
    else:
        print(f"Latent output shape: {video.shape}")

    return video


def test_wan_tt_pipeline():
    """Test Wan TT pipeline: 2 denoising steps, latent output only."""
    xr.set_device_type("TT")

    output_path = "test_wan_tt_output.mp4"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    try:
        video = run_wan_tt_pipeline(
            output_path=output_path,
            num_inference_steps=2,
            output_type="latent",
        )
        assert video is not None, "Pipeline returned None"
        print(f"Output shape: {video.shape}")
        print("Wan TT pipeline test passed.")
    finally:
        if output_file.exists():
            output_file.unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    test_wan_tt_pipeline()
