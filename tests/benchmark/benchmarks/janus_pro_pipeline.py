# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Janus-Pro-1B — benchmark-side pipeline for the AR image-gen harness.

Self-contained (no import from ``examples/`` or the nightly test): the
language_model.model + gen_head step, gen_img_embed and gen_vision_decode each
move to Tenstorrent via ``model.compile(backend="tt") + model.to(xla_device())``
and are evicted back to CPU after use. The autoregressive step uses a
``StaticCache`` pre-allocated to ``prompt_len + num_image_tokens`` so prefill and
decode compile once and the rest of the loop are cache hits.

Per-stage forward+sync times are collected into ``self._perf`` for the harness:
``prefill`` (s), ``decode_steps`` (list of per-step s), ``vision_decode`` (s),
``total`` (s). A ``.to("cpu")`` after each forward forces the XLA sync so the
timers bracket real device work.
"""

import time
from typing import Optional

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
from loguru import logger
from transformers import StaticCache

from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import model_utils
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src.model import (
    JanusGenImgEmbed,
    JanusGenVisionDecode,
)

REPO_ID_PRO_1B = model_utils.REPO_ID_PRO_1B
REPO_ID_PRO_7B = model_utils.REPO_ID_PRO_7B
MODEL_ID = REPO_ID_PRO_1B
CFG_WEIGHT = 5.0
TEMPERATURE = 1.0
IMG_SIZE = 384
PARALLEL_SIZE = 1
CFG_BATCH = PARALLEL_SIZE * 2
DTYPE = torch.bfloat16


class JanusImageTokenStep(nn.Module):
    """language_model.model + gen_head, StaticCache-aware (pre-CFG logits)."""

    def __init__(self, mmgpt):
        super().__init__()
        self.lm = mmgpt.language_model.model
        self.gen_head = mmgpt.gen_head

    def forward(
        self,
        inputs_embeds,
        cache_position,
        position_ids,
        attention_mask,
        past_key_values,
    ):
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
        image_token_on_tt: bool = True,
        gen_img_embed_on_tt: bool = True,
        gen_vision_decode_on_tt: bool = True,
        compile_options: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.image_token_on_tt = image_token_on_tt
        self.gen_img_embed_on_tt = gen_img_embed_on_tt
        self.gen_vision_decode_on_tt = gen_vision_decode_on_tt
        # Harness-set compile options (forwarded for inline merges if needed).
        self.compile_options = compile_options or {}


class JanusProPipeline:
    """Janus-Pro-1B AR text-to-image pipeline with per-stage perf timing."""

    def __init__(self, config: JanusProConfig):
        self.config = config

    def setup(self):
        self.processor = model_utils.load_processor(self.config.model_id)
        self.mmgpt = model_utils.load_mmgpt(self.config.model_id, DTYPE)

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

    def _sample(self, logits):
        cond = logits[0::2, :]
        uncond = logits[1::2, :]
        guided = uncond + CFG_WEIGHT * (cond - uncond)
        probs = torch.softmax(guided.float() / TEMPERATURE, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate(
        self,
        prompt: str,
        num_image_tokens: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)

        cfg = self.config
        device = xm.xla_device()
        tt_cast = lambda x: x.to(device=device)
        cpu_cast = lambda x: x.to("cpu")
        step_on_tt = cfg.image_token_on_tt
        embed_on_tt = cfg.gen_img_embed_on_tt

        self._perf = {
            "prefill": None,
            "decode_steps": [],
            "vision_decode": None,
            "total": None,
        }
        t_total = time.perf_counter()

        with torch.no_grad():
            full_prompt = model_utils.build_prompt(self.processor, prompt)
            prefill_embeds = model_utils.prepare_cfg_inputs_embeds(
                self.mmgpt,
                self.processor,
                full_prompt,
                parallel_size=PARALLEL_SIZE,
                device="cpu",
            ).to(DTYPE)
            prompt_len = prefill_embeds.shape[1]
            max_cache_len = prompt_len + num_image_tokens

            logger.info("[STAGE] Image-token AR loop: start")
            if step_on_tt:
                self.step = self.step.to(device)
            if embed_on_tt:
                self.gen_img_embed = self.gen_img_embed.to(device)
            cache = self._make_static_cache(
                max_cache_len, device if step_on_tt else "cpu"
            )
            generated = torch.zeros((PARALLEL_SIZE, num_image_tokens), dtype=torch.int)

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
                # cpu cast forces sync — caller's timer ends after this.
                return cpu_cast(logits) if step_on_tt else logits

            # Prefill (compiles the prefill graph once).
            t0 = time.perf_counter()
            logits = run_step(prefill_embeds, prompt_len, torch.arange(0, prompt_len))
            self._perf["prefill"] = time.perf_counter() - t0
            next_token = self._sample(logits)
            generated[:, 0] = next_token.squeeze(-1)
            cfg_token = torch.cat([next_token, next_token], dim=1).view(-1)

            # Decode loop (decode graph compiles on the first iteration).
            cur = prompt_len
            for i in range(1, num_image_tokens):
                t0 = time.perf_counter()
                image_ids = tt_cast(cfg_token) if embed_on_tt else cfg_token
                img_embeds = self.gen_img_embed(image_ids).unsqueeze(1).to(dtype=DTYPE)
                if embed_on_tt and not step_on_tt:
                    img_embeds = cpu_cast(img_embeds)
                logits = run_step(img_embeds, cur + 1, torch.tensor([cur]))
                self._perf["decode_steps"].append(time.perf_counter() - t0)
                next_token = self._sample(logits)
                generated[:, i] = next_token.squeeze(-1)
                cfg_token = torch.cat([next_token, next_token], dim=1).view(-1)
                cur += 1

            if step_on_tt:
                self.step = self.step.to("cpu")
            if embed_on_tt:
                self.gen_img_embed = self.gen_img_embed.to("cpu")
            logger.info("[STAGE] Image-token AR loop: done")

            logger.info("[STAGE] Vision decode: start")
            t0 = time.perf_counter()
            if cfg.gen_vision_decode_on_tt:
                self.gen_vision_decode = self.gen_vision_decode.to(device)
                image = cpu_cast(self.gen_vision_decode(tt_cast(generated)))
                self.gen_vision_decode = self.gen_vision_decode.to("cpu")
            else:
                image = self.gen_vision_decode(generated)
            self._perf["vision_decode"] = time.perf_counter() - t0
            logger.info("[STAGE] Vision decode: done")

            self._perf["total"] = time.perf_counter() - t_total
            return image
