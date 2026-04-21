# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel


class WanConfig:
    def __init__(
        self,
        device="cpu",
        encoder_on_tt=False,
        transformer_on_tt=False,
        vae_on_tt=False,
    ):
        self.model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        self.height = 480
        self.width = 832
        self.num_frames = 81
        self.device = device
        self.encoder_on_tt = encoder_on_tt
        self.transformer_on_tt = transformer_on_tt
        self.vae_on_tt = vae_on_tt


class WanT2VPipeline:
    """Pipeline for video generation with Wan 2.2 TI2V-5B."""

    def __init__(self, config: WanConfig):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id
        self.encoder_on_tt = config.encoder_on_tt
        self.transformer_on_tt = config.transformer_on_tt
        self.vae_on_tt = config.vae_on_tt

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
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16 if self.encoder_on_tt else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        if self.encoder_on_tt:
            self.text_encoder.compile(backend="tt")
            self.text_encoder = self.text_encoder.to(xm.xla_device())

        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16 if self.transformer_on_tt else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        if self.transformer_on_tt:
            self.transformer.compile(backend="tt")
            self.transformer = self.transformer.to(xm.xla_device())

        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        if self.vae_on_tt:
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

    def _encode_prompt_single(self, prompt_list, max_sequence_length):
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
        seq_lens = mask.gt(0).sum(dim=1).long()  # actual token count per sequence

        # CPU → TT
        if self.encoder_on_tt:
            text_input_ids = text_input_ids.to(xm.xla_device())
            mask = mask.to(xm.xla_device())

        embeds = self.text_encoder(text_input_ids, mask).last_hidden_state

        # TT → CPU
        if self.encoder_on_tt:
            embeds = embeds.to("cpu").to(dtype=torch.float32)
        else:
            embeds = embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        # Trim to actual seq length, then re-pad to max_sequence_length
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in embeds
            ],
            dim=0,
        )
        return embeds

    def encode_prompt(self, prompt, negative_prompt="", max_sequence_length=512):
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
            # --- Align num_frames to VAE temporal factor (must be k*factor + 1) ---
            if num_frames % self.vae_scale_factor_temporal != 1:
                num_frames = (
                    num_frames
                    // self.vae_scale_factor_temporal
                    * self.vae_scale_factor_temporal
                    + 1
                )
            num_frames = max(num_frames, 1)

            # --- Align height/width to patch and VAE spatial factor ---
            patch_size = self.transformer.config.patch_size
            h_multiple = self.vae_scale_factor_spatial * patch_size[1]
            w_multiple = self.vae_scale_factor_spatial * patch_size[2]
            height = height // h_multiple * h_multiple
            width = width // w_multiple * w_multiple

            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # --- Text encoding (UMT5) ---
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt, negative_prompt, max_sequence_length
            )

            transformer_dtype = self.transformer.dtype
            prompt_embeds = prompt_embeds.to(transformer_dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

            # --- Prepare timesteps ---
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            # --- Prepare latents ---
            num_channels_latents = self.transformer.config.in_channels
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
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

            # Spatial mask for building per-patch timestep vector
            mask = torch.ones_like(latents)

            # --- Denoising loop (Transformer) ---
            for i, t in enumerate(timesteps):
                latent_model_input = latents.to(transformer_dtype)

                # Per-patch timestep: broadcast scalar t across spatial patches
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)

                # CPU → TT
                if self.transformer_on_tt:
                    latent_model_input = tt_cast(latent_model_input)
                    timestep = tt_cast(timestep)
                    prompt_embeds_input = tt_cast(prompt_embeds)
                else:
                    prompt_embeds_input = prompt_embeds

                # Conditional forward pass
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds_input,
                        return_dict=False,
                    )[0]

                # TT → CPU
                if self.transformer_on_tt:
                    noise_pred = cpu_cast(noise_pred)

                # --- CFG: unconditional pass + blending ---
                if do_cfg:
                    if self.transformer_on_tt:
                        neg_embeds_input = tt_cast(negative_prompt_embeds)
                    else:
                        neg_embeds_input = negative_prompt_embeds

                    with self.transformer.cache_context("uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=neg_embeds_input,
                            return_dict=False,
                        )[0]

                    if self.transformer_on_tt:
                        noise_uncond = cpu_cast(noise_uncond)

                    # CFG blending (CPU)
                    noise_pred = noise_uncond + guidance_scale * (
                        noise_pred - noise_uncond
                    )

                # Scheduler step (CPU)
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                print(f"  Step {i + 1}/{num_inference_steps}")

            # --- VAE decode ---
            if output_type != "latent":
                latents_vae = latents.to(dtype=torch.float32)

                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents_vae.device, latents_vae.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents_vae.device, latents_vae.dtype)
                latents_vae = latents_vae / latents_std + latents_mean

                # CPU → TT
                if self.vae_on_tt:
                    latents_vae = latents_vae.to(device=xm.xla_device())

                video = self.vae.decode(latents_vae, return_dict=False)[0]

                # TT → CPU
                if self.vae_on_tt:
                    video = cpu_cast(video)

                video = self.video_processor.postprocess_video(
                    video, output_type=output_type
                )
            else:
                video = latents

            return video


def run_wan_pipeline(
    output_path: str = "wan_output.mp4",
    num_inference_steps: int = 50,
    output_type: str = "np",
):
    torch_xla.set_custom_compile_options({"optimization_level": 1})

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
        export_to_video(video, output_path, fps=24)
        print(f"Video saved to {output_path}")
    else:
        print(f"Latent output shape: {video.shape}")

    return video


def test_wan_pipeline():
    """Test Wan pipeline: 2 denoising steps, latent output only."""
    xr.set_device_type("TT")

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
