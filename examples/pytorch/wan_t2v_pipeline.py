# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU-only Wan video generation pipeline for validation against diffusers WanPipeline.

Replicates the forward pass of:
  diffusers.pipelines.wan.pipeline_wan.WanPipeline.__call__

in the same style as sd_v1_4_pipeline.py replicates StableDiffusionPipeline.
All computation is float32 on CPU. No TT/XLA code.
"""

import html
import re
from pathlib import Path
from typing import Optional

import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
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


class WanConfig:
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        expand_timesteps: bool = True,
    ):
        self.model_id = model_id
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.device = device
        self.dtype = dtype
        self.expand_timesteps = expand_timesteps


class WanT2VPipeline:
    """CPU-only pipeline for video generation with Wan.

    Mirrors diffusers WanPipeline.__call__ exactly so that outputs can be
    compared 1:1.  Later, TT compile/cast can be layered on top (same
    approach as SD14Pipeline in sd_v1_4_pipeline.py).
    """

    def __init__(self, config: WanConfig):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id
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
        dtype = self.config.dtype

        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )

        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )

        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )

    def load_scheduler(self):
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )

    # ------------------------------------------------------------------
    # Text encoding – mirrors WanPipeline._get_t5_prompt_embeds exactly
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

        embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
        embeds = embeds.to(dtype=self.text_encoder.dtype, device=self.device)

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
        """Encode prompt + negative prompt (mirrors WanPipeline.encode_prompt)."""
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
    # Generate – mirrors WanPipeline.__call__
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
        """Generate video from a text prompt. Returns frames or latent tensor."""
        height = height or self.config.height
        width = width or self.config.width
        num_frames = num_frames or self.config.num_frames
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_cfg = guidance_scale > 1.0

        with torch.no_grad():
            # --- 1. Adjust num_frames (must satisfy: (F-1) % temporal_factor == 0) ---
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

            # --- 3. Generator ---
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # --- 4. Text encoding (T5) ---
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt, negative_prompt, max_sequence_length
            )

            transformer_dtype = self.transformer.dtype
            prompt_embeds = prompt_embeds.to(transformer_dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

            # --- 5. Prepare timesteps ---
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            # --- 6. Prepare latents (5-D: B, C, T, H, W) ---
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

            mask = torch.ones_like(latents)

            # --- 7. Denoising loop ---
            for i, t in enumerate(timesteps):
                latent_model_input = latents.to(transformer_dtype)

                if self.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)
                else:
                    timestep = t.expand(batch_size)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]

                if do_cfg:
                    with self.transformer.cache_context("uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + guidance_scale * (
                        noise_pred - noise_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                print(f"  Step {i + 1}/{num_inference_steps}")

            # --- 8. VAE decode ---
            if output_type != "latent":
                latents_vae = latents.to(self.vae.dtype)
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
                video = self.vae.decode(latents_vae, return_dict=False)[0]
                video = self.video_processor.postprocess_video(
                    video, output_type=output_type
                )
            else:
                video = latents

            return video


# ======================================================================
# Convenience helpers
# ======================================================================


def run_wan_pipeline(
    output_path: str = "wan_output.mp4",
    num_inference_steps: int = 50,
    output_type: str = "np",
):
    """Run Wan pipeline on CPU and optionally save the video."""
    config = WanConfig(device="cpu")
    pipeline = WanT2VPipeline(config=config)
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


def test_wan_pipeline():
    """Quick smoke-test: 2 denoising steps, latent output only (no VAE decode)."""
    output_path = "test_wan_output.mp4"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    try:
        video = run_wan_pipeline(
            output_path=output_path,
            num_inference_steps=2,
            output_type="latent",
        )
        assert video is not None, "Pipeline returned None"
        print(f"Output shape: {video.shape}")
        print("Wan pipeline test passed.")
    finally:
        if output_file.exists():
            output_file.unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    test_wan_pipeline()
