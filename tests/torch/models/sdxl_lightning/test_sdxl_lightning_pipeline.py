# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SDXL-Lightning — nightly e2e pipeline test with per-component PCC checks.

Every nn.Module component (text_encoder, text_encoder_2, unet, vae) runs on
Tenstorrent. After each TT forward, the same real pipeline tensors are fed to
a fp32 CPU "twin" of the component and PCC is checked immediately. Test fails
fast the moment any PCC drops below `PCC_THRESHOLD`.

The pipeline itself continues with TT outputs (real deployment behavior); each
per-component PCC assertion is measured against a clean fp32 CPU reference fed
the same input the TT component saw.
"""

from typing import Optional

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import EulerDiscreteScheduler
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from transformers import CLIPTokenizer
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.sdxl_lightning.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.sdxl_lightning.pytorch.src.model_utils import (
    SDXL_BASE_REPO_ID,
)

MODEL_ID = SDXL_BASE_REPO_ID
PROMPT = "A girl smiling"
SEED = 42
NUM_INFERENCE_STEPS = 4
HEIGHT = 1024
WIDTH = 1024
PCC_THRESHOLD = 0.99


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


class SDXLLightningConfig:
    def __init__(self, device: str = "cpu"):
        self.model_id = MODEL_ID
        self.width = WIDTH
        self.height = HEIGHT
        self.vae_scale_factor = 8
        self.latents_width = self.width // self.vae_scale_factor
        self.latents_height = self.height // self.vae_scale_factor
        self.device = device


class SDXLLightningPipeline:
    """SDXL-Lightning pipeline: all four components run on TT with PCC checks."""

    def __init__(self, config: SDXLLightningConfig):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizers()

    def load_models(self):
        # TT-bound models: load on CPU, register the "tt" dynamo backend here,
        # move to xla_device inline in generate() right before the forward, then
        # evict back to CPU. Keeps at most one model resident on TT DRAM at a time.
        self.text_encoder = ModelLoader(ModelVariant.TEXT_ENCODER).load_model(
            dtype_override=torch.float32
        )
        self.text_encoder.compile(backend="tt")

        self.text_encoder_2 = ModelLoader(ModelVariant.TEXT_ENCODER_2).load_model(
            dtype_override=torch.float32
        )
        self.text_encoder_2.compile(backend="tt")

        # UNet uses bf16 on TT to fit DRAM. CPU twin below stays fp32 as golden.
        self.unet = ModelLoader(ModelVariant.UNET).load_model(
            dtype_override=torch.bfloat16
        )
        self.unet.compile(backend="tt")

        self.vae = ModelLoader(ModelVariant.VAE).load_model(
            dtype_override=torch.float32
        )
        self.vae.compile(backend="tt")

        # Lazy CPU fp32 twins for PCC comparison. Loaded on first use.
        self._cpu_twins = {}

    def _cpu_twin(self, variant: ModelVariant):
        if variant not in self._cpu_twins:
            logger.info(f"[PCC] loading CPU twin: {variant}")
            self._cpu_twins[variant] = ModelLoader(variant).load_model(
                dtype_override=torch.float32
            )
        return self._cpu_twins[variant]

    def load_scheduler(self):
        # SDXL-Lightning requires Euler with "trailing" timestep spacing.
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler", timestep_spacing="trailing"
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
        num_inference_steps: int = 4,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = 1

        tt_cast = lambda x: x.to(device=xm.xla_device())
        cpu_cast = lambda x: x.to("cpu")

        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)

            # ── Text encoder 1 (CLIPTextModel) ────────────────────────────
            logger.info("[STAGE] Text encoder 1: start")
            tokens_1 = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device=self.device)
            tokens_1_cpu = tokens_1.clone()

            self.text_encoder = self.text_encoder.to(xm.xla_device())
            tokens_1 = tokens_1.to(device=xm.xla_device())
            prompt_embeds_1 = self.text_encoder(tokens_1)
            prompt_embeds_1 = cpu_cast(prompt_embeds_1)
            self.text_encoder = self.text_encoder.to("cpu")

            golden_te1 = self._cpu_twin(ModelVariant.TEXT_ENCODER)(tokens_1_cpu)
            pcc_te1 = _pcc(prompt_embeds_1, golden_te1)
            logger.info(f"[PCC] text_encoder_1: pcc={pcc_te1:.6f}")
            assert (
                pcc_te1 >= PCC_THRESHOLD
            ), f"text_encoder_1 PCC {pcc_te1:.6f} below threshold {PCC_THRESHOLD}"

            # ── Text encoder 2 (CLIPTextModelWithProjection) ──────────────
            logger.info("[STAGE] Text encoder 2: start")
            tokens_2 = self.tokenizer_2(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device=self.device)
            tokens_2_cpu = tokens_2.clone()

            self.text_encoder_2 = self.text_encoder_2.to(xm.xla_device())
            tokens_2 = tokens_2.to(device=xm.xla_device())
            prompt_embeds_2, pooled_prompt_embeds = self.text_encoder_2(tokens_2)
            prompt_embeds_2 = cpu_cast(prompt_embeds_2)
            pooled_prompt_embeds = cpu_cast(pooled_prompt_embeds)
            self.text_encoder_2 = self.text_encoder_2.to("cpu")

            golden_te2_hidden, golden_te2_pooled = self._cpu_twin(
                ModelVariant.TEXT_ENCODER_2
            )(tokens_2_cpu)
            pcc_te2_hidden = _pcc(prompt_embeds_2, golden_te2_hidden)
            pcc_te2_pooled = _pcc(pooled_prompt_embeds, golden_te2_pooled)
            logger.info(
                f"[PCC] text_encoder_2: hidden_pcc={pcc_te2_hidden:.6f} "
                f"pooled_pcc={pcc_te2_pooled:.6f}"
            )
            assert (
                pcc_te2_hidden >= PCC_THRESHOLD
            ), f"text_encoder_2 hidden PCC {pcc_te2_hidden:.6f} below threshold {PCC_THRESHOLD}"
            assert (
                pcc_te2_pooled >= PCC_THRESHOLD
            ), f"text_encoder_2 pooled PCC {pcc_te2_pooled:.6f} below threshold {PCC_THRESHOLD}"

            # Concat the two encoders' hidden states (no CFG: batch stays 1).
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
            add_text_embeds = pooled_prompt_embeds  # (1, 1280)

            add_time_ids = self._get_add_time_ids(prompt_embeds.dtype).to(
                self.device
            )  # (1, 6)

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
            latents = torch.randn(latent_shape, dtype=torch.float32).to(
                device=self.device
            )
            latents = latents * self.scheduler.init_noise_sigma

            # ── Denoising loop (UNet, no CFG) ─────────────────────────────
            logger.info(
                f"[STAGE] UNet denoising loop: start ({num_inference_steps} steps)"
            )
            self.unet = self.unet.to(xm.xla_device())
            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] UNet step {i + 1}/{num_inference_steps}")

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # CPU → TT (UNet runs in bf16 on TT).
                unet_sample = tt_cast(latent_model_input.to(torch.bfloat16))
                unet_t = tt_cast(t.to(torch.bfloat16))
                unet_eh = tt_cast(prompt_embeds.to(torch.bfloat16))
                unet_te = tt_cast(add_text_embeds.to(torch.bfloat16))
                unet_ti = tt_cast(add_time_ids.to(torch.bfloat16))

                noise_pred = self.unet(unet_sample, unet_t, unet_eh, unet_te, unet_ti)
                noise_pred = cpu_cast(noise_pred).to(torch.float32)

                # CPU golden fed the same fp32 tensors the TT UNet consumed.
                golden_noise = self._cpu_twin(ModelVariant.UNET)(
                    latent_model_input,
                    t,
                    prompt_embeds,
                    add_text_embeds,
                    add_time_ids,
                )
                pcc_unet = _pcc(noise_pred, golden_noise)
                logger.info(
                    f"[PCC] unet step {i + 1}/{num_inference_steps}: pcc={pcc_unet:.6f}"
                )
                assert pcc_unet >= PCC_THRESHOLD, (
                    f"unet step {i + 1}/{num_inference_steps} PCC {pcc_unet:.6f} "
                    f"below threshold {PCC_THRESHOLD}"
                )

                # No CFG combine (guidance_scale=0): use noise_pred directly.
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            self.unet = self.unet.to("cpu")
            logger.info("[STAGE] UNet denoising loop: done")

            # ── VAE decode (standard SDXL: divide by scaling_factor) ──────
            logger.info("[STAGE] VAE decode: start")
            latents = latents / self.vae.vae.config.scaling_factor
            latents_cpu = latents.clone()

            # opt_level=1 (composite ttnn.group_norm) needed for VAE on TT.
            torch_xla.set_custom_compile_options({"optimization_level": 1})
            self.vae = self.vae.to(xm.xla_device())
            latents = tt_cast(latents)

            image = self.vae(latents)

            image = cpu_cast(image)
            self.vae = self.vae.to("cpu")

            golden_image = self._cpu_twin(ModelVariant.VAE)(latents_cpu)
            pcc_vae = _pcc(image, golden_image)
            logger.info(f"[PCC] vae: pcc={pcc_vae:.6f}")
            assert (
                pcc_vae >= PCC_THRESHOLD
            ), f"vae PCC {pcc_vae:.6f} below threshold {PCC_THRESHOLD}"

            logger.info("[STAGE] VAE decode: done")
            return image


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="SDXLLightning_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_sdxl_lightning_pipeline():
    """SDXL-Lightning pipeline (all components on TT) with per-component PCC checks."""
    xr.set_device_type("TT")

    config = SDXLLightningConfig(device="cpu")
    pipeline = SDXLLightningPipeline(config=config)
    pipeline.setup()
    pipeline.generate(
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
