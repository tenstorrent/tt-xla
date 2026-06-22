# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Janus-Pro (1B + 7B) — nightly e2e text-to-image pipeline test.

Mirrors the deepseek-ai/Janus ``generation_inference.py`` reference loop
(``parallel_size=1``): build CFG prompt embeds, run a 576-step autoregressive
image-token loop (language_model.model + gen_head, classifier-free guidance +
multinomial sampling), then decode the image tokens with
``gen_vision_model.decode_code`` into a 384x384 image.

The three components are the same ones the per-component bring-up tests exercise
(``test_image_token_step.py``, ``test_gen_img_embed.py``,
``test_gen_vision_decode.py``); here they are chained end-to-end. Each component
is moved to Tenstorrent via ``model.compile(backend="tt") + model.to(xla_device())``
and evicted back to CPU after use, keeping at most one model resident on TT DRAM
(the autoregressive step + gen_img_embed share the loop, then the vision decoder
takes the device). The processor/tokenizer always stay on CPU.

The autoregressive step uses a ``StaticCache`` pre-allocated to
``prompt_len + 576`` so the prefill and decode graphs compile once and every
later decode step is a cache hit (a ``DynamicCache``, as in the original script,
recompiles on each step because the KV length changes — hours on TT). Token
output is not bit-identical to CPU (bf16/PCC differences under multinomial
sampling diverge after the first step), so this test validates that a
well-formed image is produced, not exact token equality — the per-component
tests cover PCC.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra import RunMode
from loguru import logger
from PIL import Image
from transformers import StaticCache
from utils import BringupStatus, Category, ModelGroup

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import model_utils
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src.model import (
    JanusGenImgEmbed,
    JanusGenVisionDecode,
)

from . import skip_pro_7b_image_token_on_wormhole

MODEL_ID = model_utils.REPO_ID_PRO_1B
PROMPT = model_utils.STANDARD_PROMPT
SEED = 42
NUM_IMAGE_TOKENS = 576
CFG_WEIGHT = 5.0
TEMPERATURE = 1.0
IMG_SIZE = 384
PARALLEL_SIZE = 1
CFG_BATCH = PARALLEL_SIZE * 2  # CFG: conditional + unconditional rows
DTYPE = torch.bfloat16


class JanusImageTokenStep(nn.Module):
    """language_model.model + gen_head, StaticCache-aware (pre-CFG logits).

    Unlike ``JanusGitImageTokenStep`` (DynamicCache, used by the component
    test), this step takes the explicit ``cache_position`` / ``position_ids`` /
    ``attention_mask`` a ``StaticCache`` needs so prefill and decode each
    compile exactly once.
    """

    def __init__(self, mmgpt):
        super().__init__()
        self.lm = mmgpt.language_model.model
        self.gen_head = mmgpt.gen_head

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values,
    ) -> torch.Tensor:
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return self.gen_head(outputs.last_hidden_state[:, -1, :])


class JanusProConfig:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cpu",
        image_token_on_tt: bool = True,
        gen_img_embed_on_tt: bool = True,
        gen_vision_decode_on_tt: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.image_token_on_tt = image_token_on_tt
        self.gen_img_embed_on_tt = gen_img_embed_on_tt
        self.gen_vision_decode_on_tt = gen_vision_decode_on_tt


class JanusProPipeline:
    """Janus-Pro-1B text-to-image pipeline (deepseek-ai/Janus reference loop)."""

    def __init__(self, config: JanusProConfig):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id

    def setup(self):
        # Load on CPU. For TT-bound components we only register the `tt` dynamo
        # backend here; the move to xla_device happens in generate() and each is
        # evicted back to CPU after use, so at most one model is resident on TT.
        self.processor = model_utils.load_processor(self.model_id)
        self.mmgpt = model_utils.load_mmgpt(self.model_id, DTYPE)

        self.step = JanusImageTokenStep(self.mmgpt).eval()
        if self.config.image_token_on_tt:
            self.step.compile(backend="tt")

        self.gen_img_embed = JanusGenImgEmbed(
            self.mmgpt.gen_embed, self.mmgpt.gen_aligner
        ).eval()
        if self.config.gen_img_embed_on_tt:
            self.gen_img_embed.compile(backend="tt")

        self.gen_vision_decode = JanusGenVisionDecode(
            self.mmgpt.gen_vision_model, model_utils.decode_shape(PARALLEL_SIZE)
        ).eval()
        if self.config.gen_vision_decode_on_tt:
            self.gen_vision_decode.compile(backend="tt")

    def _make_static_cache(self, max_cache_len: int, device):
        cfg = self.mmgpt.language_model.config
        cache = StaticCache(
            config=cfg,
            max_batch_size=CFG_BATCH,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=DTYPE,
        )
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        # StaticCache is initialized on CPU and transferred to device separately
        # (a trace/fusion issue otherwise); see examples/pytorch/llama.py.
        cache.early_initialization(
            batch_size=CFG_BATCH,
            num_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            dtype=DTYPE,
            device="cpu",
        )
        if device != "cpu":
            for layer in cache.layers:
                layer.keys = layer.keys.to(device)
                layer.values = layer.values.to(device)
                if hasattr(layer, "cumulative_length"):
                    layer.cumulative_length = layer.cumulative_length.to(device)
                layer.device = device
        return cache

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """CFG-combine pre-CFG logits and multinomial-sample one token per image."""
        cond = logits[0::2, :]
        uncond = logits[1::2, :]
        guided = uncond + CFG_WEIGHT * (cond - uncond)
        probs = torch.softmax(guided.float() / TEMPERATURE, dim=-1)
        return torch.multinomial(probs, num_samples=1)  # [PARALLEL_SIZE, 1]

    def generate(
        self,
        prompt: str,
        num_image_tokens: int = NUM_IMAGE_TOKENS,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)

        cfg = self.config
        tt_cast = lambda x: x.to(device=xm.xla_device())
        cpu_cast = lambda x: x.to("cpu")

        with torch.no_grad():
            # CFG prompt embeds are built on CPU (text embed_tokens runs on CPU,
            # before the language model is moved to TT).
            full_prompt = model_utils.build_prompt(self.processor, prompt)
            prefill_embeds = model_utils.prepare_cfg_inputs_embeds(
                self.mmgpt,
                self.processor,
                full_prompt,
                parallel_size=PARALLEL_SIZE,
                device=self.device,
            ).to(DTYPE)
            prompt_len = prefill_embeds.shape[1]
            max_cache_len = prompt_len + num_image_tokens
            logger.info(
                f"[Janus-Pro] prompt_len={prompt_len} "
                f"num_image_tokens={num_image_tokens} max_cache_len={max_cache_len}"
            )

            step_on_tt = cfg.image_token_on_tt
            embed_on_tt = cfg.gen_img_embed_on_tt
            generated = torch.zeros((PARALLEL_SIZE, num_image_tokens), dtype=torch.int)

            # ── Image-token AR loop (language_model.model + gen_head) ──────
            logger.info("[STAGE] Image-token AR loop: start")
            if step_on_tt:
                self.step = self.step.to(xm.xla_device())
            if embed_on_tt:
                self.gen_img_embed = self.gen_img_embed.to(xm.xla_device())
            cache = self._make_static_cache(
                max_cache_len, xm.xla_device() if step_on_tt else "cpu"
            )

            def run_step(inputs_embeds, valid_len, cache_position):
                position_ids = cache_position.unsqueeze(0).expand(CFG_BATCH, -1)
                attention_mask = torch.zeros(
                    (CFG_BATCH, max_cache_len), dtype=torch.long
                )
                attention_mask[:, :valid_len] = 1
                if step_on_tt:
                    inputs_embeds = tt_cast(inputs_embeds)
                    cache_position = tt_cast(cache_position)
                    position_ids = tt_cast(position_ids)
                    attention_mask = tt_cast(attention_mask)
                logits = self.step(
                    inputs_embeds, cache_position, position_ids, attention_mask, cache
                )
                return cpu_cast(logits) if step_on_tt else logits

            # Prefill (compiles the prefill graph once).
            logits = run_step(prefill_embeds, prompt_len, torch.arange(0, prompt_len))
            next_token = self._sample(logits)
            generated[:, 0] = next_token.squeeze(-1)
            cfg_token = torch.cat([next_token, next_token], dim=1).view(-1)

            # Decode loop (compiles the decode graph once; rest are cache hits).
            cur = prompt_len
            for i in range(1, num_image_tokens):
                image_ids = tt_cast(cfg_token) if embed_on_tt else cfg_token
                img_embeds = self.gen_img_embed(image_ids).unsqueeze(1).to(dtype=DTYPE)
                if embed_on_tt and not step_on_tt:
                    img_embeds = cpu_cast(img_embeds)
                logits = run_step(img_embeds, cur + 1, torch.tensor([cur]))
                next_token = self._sample(logits)
                generated[:, i] = next_token.squeeze(-1)
                cfg_token = torch.cat([next_token, next_token], dim=1).view(-1)
                cur += 1
                if i % 128 == 0:
                    logger.info(f"[STEP] AR loop {i}/{num_image_tokens}")

            # Evict the AR components before the vision decoder takes the device.
            if step_on_tt:
                self.step = self.step.to("cpu")
            if embed_on_tt:
                self.gen_img_embed = self.gen_img_embed.to("cpu")
            logger.info("[STAGE] Image-token AR loop: done")

            # ── Vision decode (gen_vision_model.decode_code) ───────────────
            logger.info("[STAGE] Vision decode: start")
            if cfg.gen_vision_decode_on_tt:
                self.gen_vision_decode = self.gen_vision_decode.to(xm.xla_device())
                image = cpu_cast(self.gen_vision_decode(tt_cast(generated)))
                self.gen_vision_decode = self.gen_vision_decode.to("cpu")
            else:
                image = self.gen_vision_decode(generated)
            logger.info("[STAGE] Vision decode: done")

            return image  # [PARALLEL_SIZE, 3, IMG_SIZE, IMG_SIZE], range [-1, 1]


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)


def run_janus_pro_pipeline(
    output_path: str = "janus_pro_output.png",
    num_image_tokens: int = NUM_IMAGE_TOKENS,
    model_id: str = MODEL_ID,
):
    """Run the Janus-Pro pipeline for ``model_id`` and save the output image."""
    config = JanusProConfig(model_id=model_id, device="cpu")
    pipeline = JanusProPipeline(config=config)
    pipeline.setup()

    img = pipeline.generate(
        prompt=PROMPT,
        num_image_tokens=num_image_tokens,
        seed=SEED,
    )

    save_image(img, output_path)
    return output_path


def _run_pipeline_test(model_id: str, output_path: str):
    """Shared body: run the pipeline for ``model_id`` and assert a 384x384 image.

    The ``janus`` runtime package is installed into the env for the duration of
    the run; the language model architecture (layer/head counts that size the
    StaticCache) is read from the loaded config, so the same pipeline serves
    both the 1B and 7B variants. Pro-7B requires blackhole (p150) — it OOMs the
    DRAM on wormhole (n150).
    """
    xr.set_device_type("TT")

    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        output_file = Path(output_path)
        if output_file.exists():
            output_file.unlink()

        run_janus_pro_pipeline(output_path=output_path, model_id=model_id)

        assert output_file.exists(), f"Output image {output_path} was not created"
        with Image.open(output_path) as img:
            width, height = img.size
            assert width == IMG_SIZE, f"Expected width {IMG_SIZE}, got {width}"
            assert height == IMG_SIZE, f"Expected height {IMG_SIZE}, got {height}"
        logger.info(f"Output image saved to {output_path} ({width}x{height})")


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="JanusPro1B_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_janus_pro_pipeline():
    """Run the full Janus-Pro-1B text-to-image pipeline on TT."""
    _run_pipeline_test(model_utils.REPO_ID_PRO_1B, "janus_pro_1b_output.png")


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="JanusPro7B_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_janus_pro_pipeline_7b():
    """Run the full Janus-Pro-7B text-to-image pipeline on TT (blackhole).

    Skips on wormhole (n150): the 7B model OOMs the DRAM there, same as the
    Pro-7B ImageTokenStep component test. Requires blackhole (p150).
    """
    skip_pro_7b_image_token_on_wormhole()
    _run_pipeline_test(model_utils.REPO_ID_PRO_7B, "janus_pro_7b_output.png")
