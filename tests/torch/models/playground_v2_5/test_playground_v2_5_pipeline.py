# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Playground v2.5 — nightly e2e pipeline test.

Each nn.Module component (text_encoder, text_encoder_2, unet, vae) can be
independently moved to Tenstorrent via
`model.compile(backend="tt") + model.to(xla_device())`. Tokenizer and
scheduler always stay on CPU. CPU→TT→CPU device switching is done inline at
the call site of each nn component.
"""

from pathlib import Path
from typing import Optional

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import EDMDPMSolverMultistepScheduler
from infra import RunMode
from loguru import logger
from PIL import Image
from transformers import CLIPTokenizer
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.playground_v2_5.pytorch import (
    ModelLoader,
    ModelVariant,
)

MODEL_ID = "playgroundai/playground-v2.5-1024px-aesthetic"
PROMPT = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
NEGATIVE_PROMPT = None
SEED = 42
GUIDANCE_SCALE = 3.0
HEIGHT = 1024
WIDTH = 1024


class PlaygroundV25Config:
    def __init__(
        self,
        device: str = "cpu",
        text_encoder_on_tt: bool = True,
        text_encoder_2_on_tt: bool = True,
        unet_on_tt: bool = True,
        vae_on_tt: bool = True,
    ):
        self.model_id = MODEL_ID
        self.width = WIDTH
        self.height = HEIGHT
        self.vae_scale_factor = 8
        self.latents_width = self.width // self.vae_scale_factor
        self.latents_height = self.height // self.vae_scale_factor
        self.device = device
        self.text_encoder_on_tt = text_encoder_on_tt
        self.text_encoder_2_on_tt = text_encoder_2_on_tt
        self.unet_on_tt = unet_on_tt
        self.vae_on_tt = vae_on_tt


class PlaygroundV25Pipeline:
    """Playground v2.5 pipeline with per-component TT toggles."""

    def __init__(self, config: PlaygroundV25Config):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizers()

    def load_models(self):
        # Load all models on CPU. For TT-bound components we only register the
        # `tt` dynamo backend here; the actual move to xla_device happens in
        # generate() right before the forward, and we evict back to CPU after.
        # This keeps at most one model resident on TT DRAM at a time.
        self.text_encoder = ModelLoader(ModelVariant.TEXT_ENCODER).load_model(
            dtype_override=torch.float32
        )
        if self.config.text_encoder_on_tt:
            self.text_encoder.compile(backend="tt")

        self.text_encoder_2 = ModelLoader(ModelVariant.TEXT_ENCODER_2).load_model(
            dtype_override=torch.float32
        )
        if self.config.text_encoder_2_on_tt:
            self.text_encoder_2.compile(backend="tt")

        # UNet on fp32 throws OOM on the second iteration of the denoising loop,
        # so UNet runs in bf16.
        unet_dtype = torch.bfloat16 if self.config.unet_on_tt else torch.float32
        self.unet = ModelLoader(ModelVariant.UNET).load_model(dtype_override=unet_dtype)
        if self.config.unet_on_tt:
            self.unet.compile(backend="tt")

        self.vae = ModelLoader(ModelVariant.VAE).load_model(
            dtype_override=torch.float32
        )
        if self.config.vae_on_tt:
            self.vae.compile(backend="tt")

    def load_scheduler(self):
        self.scheduler = EDMDPMSolverMultistepScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

    def load_tokenizers(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer_2"
        )

    def _get_add_time_ids(self, dtype):
        original_size = (self.config.height, self.config.width)
        crops_coords_top_left = (0, 0)
        target_size = (self.config.height, self.config.width)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([add_time_ids], dtype=dtype)

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        cfg_scale: float = 3.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = 1

        tt_cast = lambda x: x.to(device=xm.xla_device())
        cpu_cast = lambda x: x.to("cpu")

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # ── Text encoder 1 (CLIPTextModel) ────────────────────────────
            logger.info("[STAGE] Text encoder 1: start")
            tokens_1 = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device=self.device)

            # CPU → TT
            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to(xm.xla_device())
                tokens_1 = tokens_1.to(device=xm.xla_device())

            prompt_embeds_1 = self.text_encoder(tokens_1)

            # TT → CPU
            if self.config.text_encoder_on_tt:
                prompt_embeds_1 = cpu_cast(prompt_embeds_1)

            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to("cpu")

            logger.info("[STAGE] Text encoder 1: done")

            # ── Text encoder 2 (CLIPTextModelWithProjection) ──────────────
            logger.info("[STAGE] Text encoder 2: start")
            tokens_2 = self.tokenizer_2(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device=self.device)

            # CPU → TT
            if self.config.text_encoder_2_on_tt:
                self.text_encoder_2 = self.text_encoder_2.to(xm.xla_device())
                tokens_2 = tokens_2.to(device=xm.xla_device())

            prompt_embeds_2, pooled_prompt_embeds = self.text_encoder_2(tokens_2)

            # TT → CPU
            if self.config.text_encoder_2_on_tt:
                prompt_embeds_2 = cpu_cast(prompt_embeds_2)
                pooled_prompt_embeds = cpu_cast(pooled_prompt_embeds)

            if self.config.text_encoder_2_on_tt:
                self.text_encoder_2 = self.text_encoder_2.to("cpu")

            logger.info("[STAGE] Text encoder 2: done")

            # Concat the two encoders' hidden states
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

            # force_zeros_for_empty_prompt=True for playground-v2.5: zero path
            # only when negative_prompt is None.
            if negative_prompt is None:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            else:
                # Encode negative prompt through both text encoders (mirrors
                # the positive flow above).
                neg_tokens_1 = self.tokenizer(
                    [negative_prompt],
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device=self.device)
                if self.config.text_encoder_on_tt:
                    neg_tokens_1 = neg_tokens_1.to(device=xm.xla_device())
                negative_prompt_embeds_1 = self.text_encoder(neg_tokens_1)
                if self.config.text_encoder_on_tt:
                    negative_prompt_embeds_1 = cpu_cast(negative_prompt_embeds_1)

                neg_tokens_2 = self.tokenizer_2(
                    [negative_prompt],
                    padding="max_length",
                    max_length=self.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device=self.device)
                if self.config.text_encoder_2_on_tt:
                    neg_tokens_2 = neg_tokens_2.to(device=xm.xla_device())
                negative_prompt_embeds_2, negative_pooled_prompt_embeds = (
                    self.text_encoder_2(neg_tokens_2)
                )
                if self.config.text_encoder_2_on_tt:
                    negative_prompt_embeds_2 = cpu_cast(negative_prompt_embeds_2)
                    negative_pooled_prompt_embeds = cpu_cast(
                        negative_pooled_prompt_embeds
                    )

                negative_prompt_embeds = torch.cat(
                    [negative_prompt_embeds_1, negative_prompt_embeds_2], dim=-1
                )

            # CFG concat (uncond first)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

            add_time_ids = self._get_add_time_ids(prompt_embeds.dtype)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(
                self.device
            )

            # ── Timesteps ─────────────────────────────────────────────────
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            # ── Latents ───────────────────────────────────────────────────
            latent_shape = (
                batch_size,
                4,
                self.config.latents_height,
                self.config.latents_width,
            )
            latents = torch.randn(
                latent_shape, generator=generator, dtype=torch.float32
            ).to(device=self.device)
            latents = latents * self.scheduler.init_noise_sigma

            # ── Denoising loop (UNet) ─────────────────────────────────────
            logger.info(
                f"[STAGE] UNet denoising loop: start ({num_inference_steps} steps)"
            )
            # Move UNet to TT once before the loop; evict after.
            if self.config.unet_on_tt:
                self.unet = self.unet.to(xm.xla_device())
            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] UNet step {i + 1}/{num_inference_steps}")

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # CPU → TT (UNet runs in bf16 on TT)
                if self.config.unet_on_tt:
                    unet_sample = tt_cast(latent_model_input.to(torch.bfloat16))
                    unet_t = tt_cast(t.to(torch.bfloat16))
                    unet_eh = tt_cast(prompt_embeds.to(torch.bfloat16))
                    unet_te = tt_cast(add_text_embeds.to(torch.bfloat16))
                    unet_ti = tt_cast(add_time_ids.to(torch.bfloat16))
                else:
                    unet_sample = latent_model_input
                    unet_t = t
                    unet_eh = prompt_embeds
                    unet_te = add_text_embeds
                    unet_ti = add_time_ids

                noise_pred = self.unet(unet_sample, unet_t, unet_eh, unet_te, unet_ti)

                # TT → CPU
                if self.config.unet_on_tt:
                    noise_pred = cpu_cast(noise_pred).to(torch.float32)

                # CFG + scheduler step
                uncond, text = noise_pred.chunk(2)
                noise_pred = uncond + cfg_scale * (text - uncond)

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            # Evict UNet from TT before VAE comes onto the device.
            if self.config.unet_on_tt:
                self.unet = self.unet.to("cpu")
            logger.info("[STAGE] UNet denoising loop: done")

            # ── VAE decode ────────────────────────────────────────────────
            logger.info("[STAGE] VAE decode: start")
            latents_mean = (
                torch.tensor(self.vae.vae.config.latents_mean)
                .view(1, 4, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.vae.config.latents_std)
                .view(1, 4, 1, 1)
                .to(latents.device, latents.dtype)
            )
            scaling_factor = self.vae.vae.config.scaling_factor
            latents = latents * latents_std / scaling_factor + latents_mean

            # opt_level=1 keeps ttir.group_norm -> ttnn.group_norm; opt_level=0
            # decomposes GroupNorm which can OOM on the VAE.
            # See: https://github.com/tenstorrent/tt-xla/issues/4710
            if self.config.vae_on_tt:
                torch_xla.set_custom_compile_options({"optimization_level": 1})

            # CPU → TT
            if self.config.vae_on_tt:
                self.vae = self.vae.to(xm.xla_device())
                latents = tt_cast(latents)

            image = self.vae(latents)

            # TT → CPU
            if self.config.vae_on_tt:
                image = cpu_cast(image)

            if self.config.vae_on_tt:
                self.vae = self.vae.to("cpu")

            logger.info("[STAGE] VAE decode: done")

            return image


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)


def run_playground_v25_pipeline(
    output_path: str = "playground_v25_output.png",
    num_inference_steps: int = 50,
):
    """Run Playground v2.5 pipeline and save output image."""
    config = PlaygroundV25Config(device="cpu")
    pipeline = PlaygroundV25Pipeline(config=config)
    pipeline.setup()

    img = pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        cfg_scale=GUIDANCE_SCALE,
        num_inference_steps=num_inference_steps,
        seed=SEED,
    )

    save_image(img, output_path)
    return output_path


@pytest.mark.xfail(
    reason=(
        "VAE compile TT_FATAL during warmup (cores harvested / device_hash mismatch), "
        "possibly due to recent uplift — "
        "https://github.com/tenstorrent/tt-xla/issues/5176"
    ),
    strict=False,
)
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="PlaygroundV2_5_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_playground_v25_pipeline():
    """Run the full Playground v2.5 pipeline with all components on TT."""
    xr.set_device_type("TT")

    output_path = "playground_v2_5_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_playground_v25_pipeline(
        output_path=output_path,
        num_inference_steps=50,
    )

    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"

    logger.info(f"Output image saved to {output_path} ({width}x{height})")
