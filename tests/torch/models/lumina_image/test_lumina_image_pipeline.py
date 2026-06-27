# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lumina-Image-2.0 — nightly e2e pipeline test.

Lumina-Image-2.0 is a flow-matching text-to-image diffusion model, so a
generation is a Python loop over a flow-match timestep schedule: one (or two,
for classifier-free guidance) transformer forwards per step, the scheduler
step, then a single VAE decode. This reimplements ``Lumina2Pipeline.__call__``
here with every learned component resident on Tenstorrent:

  - Gemma-2 text encoder runs on TT (sharded).
  - Lumina2Transformer2DModel (DiT) runs on TT (sharded), invoked twice per
    denoising step for CFG.
  - AutoencoderKL decoder runs on TT (sharded) for the final latent->image.

All three are 2-D-mesh sharded ("batch", "model"), single-axis Megatron tensor
parallelism on the "model" axis (see ``ModelLoader.load_shard_spec`` /
``model_utils.shard_*_specs``). The tokenizer, scheduler, latent sampling,
classifier-free guidance combination and image post-processing stay on CPU --
only per-stage activations cross the CPU<->TT boundary.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import transformers.masking_utils as hf_masking_utils
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.lumina2.pipeline_lumina2 import (
    calculate_shift,
    retrieve_timesteps,
)
from infra import RunMode
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.lumina_image.pytorch import ModelLoader, ModelVariant

PROMPT = "A fantasy landscape with mountains and rivers"
NEGATIVE_PROMPT = ""
SEED = 42
HEIGHT = 1024
WIDTH = 1024
# Number of denoising steps. Lumina2Pipeline defaults to 30; lower it for a
# quicker run. Each step runs the transformer twice (CFG cond + uncond).
# Start at 1 for an initial smoke test that just confirms an image is saved;
# raise to 30 for a real generation.
NUM_INFERENCE_STEPS = 1
GUIDANCE_SCALE = 4.0
# Lumina2Pipeline default system prompt, prepended to every user prompt.
SYSTEM_PROMPT = (
    "You are an assistant designed to generate superior images with the superior "
    "degree of image-text alignment based on textual prompts or user prompts."
)
MAX_SEQUENCE_LENGTH = 256
VAE_SCALE_FACTOR = 8


@contextmanager
def _force_hf_mask_tracing():
    """Make transformers' attention-mask helpers take their ``is_tracing`` path.

    Gemma-2 builds its causal mask via ``transformers.masking_utils``, whose
    helpers (``_ignore_causal_mask_sdpa``, ``find_packed_sequence_indices``) run
    data-dependent ``.all()`` checks inside plain Python ``if`` statements --
    gated on ``not is_tracing(tensor)``. ``is_tracing`` only detects dynamo /
    jit / fx / fake-tensor, NOT torch_xla's lazy tensors, so on TT those ``.all()``
    reductions execute on device and lower to mesh-distributed ``prod`` ops the
    multichip runtime cannot collapse to a single buffer (error code 13 /
    "Can't get a single buffer ... distributed over mesh shape [1, 8]").

    Forcing ``is_tracing`` True makes these helpers skip the data-dependent
    branches and emit a fully static causal mask (single-sequence packed mask of
    zeros, broadcast-built 4D mask) -- the correct behavior for a graph-captured
    backend like torch_xla. Scoped to the encoder forward only.
    """
    with patch.object(hf_masking_utils, "is_tracing", lambda *a, **k: True):
        yield


class LuminaImageConfig:
    def __init__(
        self,
        on_tt: bool = True,
        shard: bool = True,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
    ):
        self.on_tt = on_tt
        # 2-D-mesh single-axis Megatron tensor-parallel sharding (see
        # model_utils.shard_*_specs). Needed so the large transformer/VAE
        # activations do not OOM on a single device. Set False to run each
        # component unsharded on one TT device (only viable at tiny sizes).
        self.shard = shard
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1


class LuminaImagePipeline:
    """Lumina-Image-2.0 pipeline: text encoder, transformer and VAE on TT."""

    def __init__(self, config: LuminaImageConfig):
        self.config = config

    def setup(self):
        self.load_models()
        if self.config.on_tt:
            if self.config.shard:
                self.shard_to_tt()
            else:
                # Unsharded: just move each component to a single TT device.
                dev = xm.xla_device()
                self.text_encoder = self.text_encoder.to(dev)
                self.transformer = self.transformer.to(dev)
                self.vae = self.vae.to(dev)

    def load_models(self):
        # Each Lumina-Image-2.0 component is an independently loadable variant of
        # the same repo; load_model returns a plain-tensor-forward wrapper.
        self.te_loader = ModelLoader(ModelVariant.TEXT_ENCODER)
        self.tf_loader = ModelLoader(ModelVariant.TRANSFORMER)
        self.vae_loader = ModelLoader(ModelVariant.VAE)

        self.text_encoder = self.te_loader.load_model(dtype_override=torch.bfloat16)
        self.transformer = self.tf_loader.load_model(dtype_override=torch.bfloat16)
        self.vae = self.vae_loader.load_model(dtype_override=torch.bfloat16)
        self.model_dtype = torch.bfloat16

        repo_id = self.te_loader._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
        self.tokenizer.padding_side = "right"
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            repo_id, subfolder="scheduler"
        )

    def shard_to_tt(self):
        # Enable SPMD, build the ("batch", "model") mesh once, then move each
        # component to the XLA device and mark every weight in its shard spec
        # (mirrors the runtime sharding the graph tester does). The three specs
        # all target the same mesh; weights stay resident + sharded for the
        # whole run while only activations cross the CPU<->TT boundary.
        enable_spmd()
        mesh_shape, mesh_names = self.te_loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        self.mesh = get_mesh(mesh_shape, mesh_names)

        dev = xm.xla_device()
        self.text_encoder = self.text_encoder.to(dev)
        self.transformer = self.transformer.to(dev)
        self.vae = self.vae.to(dev)
        for model, loader in (
            (self.text_encoder, self.te_loader),
            (self.transformer, self.tf_loader),
            (self.vae, self.vae_loader),
        ):
            for tensor, spec in loader.load_shard_spec(model).items():
                xs.mark_sharding(tensor, self.mesh, spec)

    # ── per-stage CPU<->TT casts (no-ops when running on CPU) ──────────
    def _tt(self, x):
        return x.to(device=xm.xla_device()) if self.config.on_tt else x

    def _cpu(self, x):
        return x.to("cpu") if self.config.on_tt else x

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> tuple:
        """Gemma-2 text encode on TT -> (prompt_embeds, attention_mask) on CPU.

        Mirrors ``Lumina2Pipeline._get_gemma_prompt_embeds``: prepend the system
        prompt, tokenize to a fixed length, run the encoder and take the
        second-to-last hidden state. ``use_cache=False`` is required to avoid the
        Gemma-2 sliding-window cache slice that exceeds tt-mlir's slice bound
        (tenstorrent/tt-xla#4900); a single encode pass needs no cache.

        The encoder forward is wrapped in ``_force_hf_mask_tracing`` so Gemma-2's
        mask helpers emit a static causal mask instead of running data-dependent
        ``.all()`` reductions on device (see that context manager). The real 2D
        padding mask is passed (not ``None``): with ``attention_mask=None``,
        ``_preprocess_mask_arguments`` would call ``find_packed_sequence_indices``,
        whose int ``cumsum`` lowers to a tt-metal accumulation kernel that rejects
        the bool->int64 dtype ("Unsupported data format for add_int"). Passing the
        mask skips that packed-sequence branch entirely, and with tracing forced
        the helper builds the padding+causal 4D mask from elementwise ops only.
        """
        full_prompt = SYSTEM_PROMPT + " <Prompt Start> " + prompt
        text_inputs = self.tokenizer(
            full_prompt,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = self._tt(text_inputs.input_ids)
        attention_mask = text_inputs.attention_mask
        with _force_hf_mask_tracing():
            out = self.text_encoder.encoder(
                input_ids=input_ids,
                attention_mask=self._tt(attention_mask),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        prompt_embeds = self._cpu(out.hidden_states[-2]).to(self.model_dtype)
        return prompt_embeds, attention_mask

    @torch.no_grad()
    def generate(self, prompt: str, seed: Optional[int] = SEED) -> torch.Tensor:
        """Reimplements ``Lumina2Pipeline.__call__`` with a CPU/TT split.

        Device map:
          - Gemma-2 text encode            -> TT
          - transformer denoising forwards -> TT (twice per step for CFG)
          - scheduler step + CFG combine   -> CPU
          - AutoencoderKL decode           -> TT
        """
        cfg = self.config
        do_cfg = cfg.do_classifier_free_guidance

        # ── text encode (TT) ──────────────────────────────────────────
        logger.info("[STAGE] Gemma-2 text encode (TT): start")
        prompt_embeds, prompt_mask = self._encode_prompt(prompt)
        if do_cfg:
            neg_embeds, neg_mask = self._encode_prompt(NEGATIVE_PROMPT)
        logger.info("[STAGE] Gemma-2 text encode (TT): done")

        # Pre-cast the (constant) conditioning to TT once, reused every step.
        prompt_embeds_tt = self._tt(prompt_embeds)
        prompt_mask_tt = self._tt(prompt_mask)
        if do_cfg:
            neg_embeds_tt = self._tt(neg_embeds)
            neg_mask_tt = self._tt(neg_mask)

        # ── prepare latents (CPU) ─────────────────────────────────────
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        latent_channels = self.transformer.transformer.config.in_channels
        # VAE applies 8x compression; latent h/w must also be divisible by 2.
        latent_h = 2 * (cfg.height // (VAE_SCALE_FACTOR * 2))
        latent_w = 2 * (cfg.width // (VAE_SCALE_FACTOR * 2))
        latents = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            generator=generator,
            dtype=self.model_dtype,
        )

        # ── timesteps (CPU, flow-match scheduler) ─────────────────────
        sigmas = np.linspace(1.0, 1 / cfg.num_inference_steps, cfg.num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, cfg.num_inference_steps, "cpu", sigmas=sigmas, mu=mu
        )
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        # ── denoising loop (transformer on TT, scheduler on CPU) ──────
        for i, t in enumerate(timesteps):
            logger.info(
                f"[STEP] denoise {i + 1}/{num_inference_steps} (t={float(t):.4f})"
            )
            # Lumina uses t=0 as noise and t=1 as image, so reverse the timestep.
            current_timestep = 1 - t / num_train_timesteps
            current_timestep = current_timestep.expand(latents.shape[0])
            ts_tt = self._tt(current_timestep)
            latents_tt = self._tt(latents)

            noise_pred_cond = self._cpu(
                self.transformer(latents_tt, ts_tt, prompt_embeds_tt, prompt_mask_tt)
            ).float()

            if do_cfg:
                noise_pred_uncond = self._cpu(
                    self.transformer(latents_tt, ts_tt, neg_embeds_tt, neg_mask_tt)
                ).float()
                noise_pred = noise_pred_uncond + cfg.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                # normalization-based guidance (cfg_normalization default True)
                cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_pred = noise_pred * (cond_norm / noise_norm)
            else:
                noise_pred = noise_pred_cond

            # x_t -> x_{t-1}. Lumina negates the predicted velocity before step.
            noise_pred = -noise_pred
            latents = self.scheduler.step(
                noise_pred.to(latents.dtype), t, latents, return_dict=False
            )[0]

        # ── VAE decode (TT) -> RGB image ──────────────────────────────
        logger.info("[STAGE] VAE decode (TT): start")
        latents = (
            latents / self.vae.vae.config.scaling_factor
        ) + self.vae.vae.config.shift_factor
        image = self._cpu(self.vae(self._tt(latents)))
        logger.info("[STAGE] VAE decode (TT): done")
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


def run_lumina_image_pipeline(
    output_path: str = "lumina_image_2_output.png",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    shard: bool = True,
):
    """Run the Lumina-Image-2.0 pipeline (all components on TT) and save image."""
    config = LuminaImageConfig(
        on_tt=True, shard=shard, num_inference_steps=num_inference_steps
    )
    pipeline = LuminaImagePipeline(config=config)
    pipeline.setup()

    img = pipeline.generate(prompt=PROMPT, seed=SEED)

    save_image(img, output_path)
    return output_path


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.tensor_parallel
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="LuminaImage2_Pipeline",
    model_group=ModelGroup.GENERALITY,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_lumina_image_2_pipeline():
    """Run the full Lumina-Image-2.0 pipeline (text encoder + transformer + VAE
    all sharded on TT, sampling/scheduler on CPU)."""
    xr.set_device_type("TT")

    output_path = "lumina_image_2_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_lumina_image_pipeline(
        output_path=output_path,
        num_inference_steps=NUM_INFERENCE_STEPS,
        shard=True,
    )

    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"

    logger.info(f"Output image saved to {output_path} ({width}x{height})")
