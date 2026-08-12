# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HiDream-I1-Full — CPU-only pipeline parity check.

Compares the stock diffusers ``HiDreamImagePipeline`` (reference) against a
manual pipeline that fetches the 6 nn.Module components via the tt-forge-models
``ModelLoader`` (the same wrappers the TT component tests use). The manual
pipeline mirrors ``HiDreamImagePipeline.__call__`` on CPU so we can confirm the
component restructuring is correct before doing the e2e TT run.

Both run on CPU in bfloat16 from the same seed, so the two images are expected
to be pixel-identical. Passing this test is what lets us attribute a later PCC
drop to the TT component swap alone.

Not wired into any CI pipeline: it loads ~60 GB of bf16 weights twice (the two
pipelines run sequentially, the first is freed before the second loads) and runs
2 x NUM_INFERENCE_STEPS forward passes of a 17 B MoE DiT on CPU — measured at
~130 s per step per pipeline on a 32-core host, i.e. ~45 min at 10 steps.
Invoke it explicitly:

    pytest -svv tests/torch/models/HiDream_I1/test_cpu_parity_check.py
"""

import gc
import math
from typing import Optional

import numpy as np
import torch
from diffusers import HiDreamImagePipeline, UniPCMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from loguru import logger
from transformers import (
    CLIPTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    T5Tokenizer,
)

from third_party.tt_forge_models.hidream_i1.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.hidream_i1.pytorch.src.model_utils import (
    HIDREAM_REPO_ID,
    LLAMA_REPO_ID,
    _patch_hidream_moe_infer,
)

MODEL_ID = HIDREAM_REPO_ID
LLAMA_ID = LLAMA_REPO_ID
# Canonical HiDream sample prompt (from the diffusers docstring / model card).
PROMPT = 'A cat holding a sign that says "HiDream.ai".'
# Left as None: HiDream's encode_prompt turns an unset negative prompt into ""
# (`negative_prompt = negative_prompt or ""`) and runs every text encoder on the
# empty string. Unlike SDXL there is no force_zeros_for_empty_prompt path here,
# so the negative branch is always four real encoder forwards.
NEGATIVE_PROMPT = None
SEED = 0
NUM_INFERENCE_STEPS = 10 # smoke run
# Per the model card: Full -> 5.0 / 50. > 1 so classifier-free guidance is on.
GUIDANCE_SCALE = 5.0
HEIGHT = 1024
WIDTH = 1024
MAX_SEQUENCE_LENGTH = 128
# HiDreamImagePipeline.default_sample_size; drives the resolution snap below.
DEFAULT_SAMPLE_SIZE = 128
# Both pipelines load in bf16 — the dtype the model actually ships in, and the
# one the CPU arch run confirmed for every component's parameters.
DTYPE = torch.bfloat16
DEVICE = "cpu"


# ── Manual pipeline (CPU mirror of HiDreamImagePipeline.__call__) ──────────


class HiDreamI1Config:
    def __init__(self, device: str = DEVICE, dtype: torch.dtype = DTYPE):
        self.model_id = MODEL_ID
        self.llama_id = LLAMA_ID
        self.height = HEIGHT
        self.width = WIDTH
        self.max_sequence_length = MAX_SEQUENCE_LENGTH
        self.default_sample_size = DEFAULT_SAMPLE_SIZE
        self.device = device
        self.dtype = dtype


class HiDreamI1Pipeline:
    """Manual HiDream pipeline using nn.Module components from tt-forge-models."""

    def __init__(self, config: HiDreamI1Config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizers()

        # HiDreamImagePipeline.__init__ derives both of these from the VAE config.
        self.vae_scale_factor = 2 ** (len(self.vae.vae.config.block_out_channels) - 1)
        # HiDream latents are turned into 2x2 patches and packed, so the image
        # processor gets twice the VAE scale factor.
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )

    def load_models(self):
        """Load the 6 components through ModelLoader — the object under test.

        Each variant returns the wrapper the TT component tests run, so the
        forward signatures here are the wrappers', not the raw HF modules'.
        """
        self.text_encoder = ModelLoader(ModelVariant.TEXT_ENCODER).load_model(
            dtype_override=self.dtype
        )
        self.text_encoder_2 = ModelLoader(ModelVariant.TEXT_ENCODER_2).load_model(
            dtype_override=self.dtype
        )
        self.text_encoder_3 = ModelLoader(ModelVariant.TEXT_ENCODER_3).load_model(
            dtype_override=self.dtype
        )
        self.text_encoder_4 = ModelLoader(ModelVariant.TEXT_ENCODER_4).load_model(
            dtype_override=self.dtype
        )
        self.transformer = ModelLoader(ModelVariant.TRANSFORMER).load_model(
            dtype_override=self.dtype
        )
        self.vae = ModelLoader(ModelVariant.VAE).load_model(dtype_override=self.dtype)

    def load_scheduler(self):
        # model_index.json pins UniPCMultistepScheduler for HiDream-I1-Full.
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.config.model_id, subfolder="scheduler"
        )

    def load_tokenizers(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.model_id, subfolder="tokenizer_2"
        )
        self.tokenizer_3 = T5Tokenizer.from_pretrained(
            self.config.model_id, subfolder="tokenizer_3"
        )
        # tokenizer_4 / text_encoder_4 are not in the HiDream snapshot — the
        # pipeline expects the caller to supply Llama-3.1-8B-Instruct.
        self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(self.config.llama_id)
        # HiDreamImagePipeline.__init__ does exactly this. Llama ships no pad
        # token, and padding="max_length" below would raise without it.
        self.tokenizer_4.pad_token = self.tokenizer_4.eos_token

    def _get_clip_prompt_embeds(self, tokenizer, text_encoder, prompt: str):
        """Mirror HiDreamImagePipeline._get_clip_prompt_embeds.

        The pipeline's untruncated-ids re-tokenization only feeds a truncation
        warning, and its trailing `.to(dtype=text_encoder.dtype)` is a no-op
        since we never override the dtype — both are dropped.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=min(self.config.max_sequence_length, 218),
            truncation=True,
            return_tensors="pt",
        )
        # CLIPPooledWrapper == text_encoder(input_ids, output_hidden_states=True)[0],
        # i.e. the projected pooled embedding, which is all the pipeline keeps.
        return text_encoder(text_inputs.input_ids.to(self.device))

    def _get_t5_prompt_embeds(self, prompt: str):
        """Mirror HiDreamImagePipeline._get_t5_prompt_embeds."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=min(
                self.config.max_sequence_length, self.tokenizer_3.model_max_length
            ),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # T5EncoderWrapper == text_encoder_3(input_ids, attention_mask=...)[0].
        return self.text_encoder_3(
            text_inputs.input_ids.to(self.device),
            text_inputs.attention_mask.to(self.device),
        )

    def _get_llama3_prompt_embeds(self, prompt: str):
        """Mirror HiDreamImagePipeline._get_llama3_prompt_embeds."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_4(
            prompt,
            padding="max_length",
            max_length=min(
                self.config.max_sequence_length, self.tokenizer_4.model_max_length
            ),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # LlamaStackedHiddenWrapper == stack(outputs.hidden_states[1:], dim=0):
        # every layer's hidden state except the embedding output -> (32,1,128,4096).
        return self.text_encoder_4(
            text_inputs.input_ids.to(self.device),
            text_inputs.attention_mask.to(self.device),
        )

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        do_classifier_free_guidance: bool = True,
    ):
        """Mirror HiDreamImagePipeline.encode_prompt for batch_size=1, 1 image/prompt.

        prompt_2/3/4 default to `prompt` and negative_prompt_2/3/4 default to
        `negative_prompt`, so one prompt drives all four encoders. The trailing
        repeat/view block in the pipeline is identity at batch_size=1 and
        num_images_per_prompt=1, so it is dropped. Call order follows the
        pipeline's (pos/neg interleaved per encoder); the encoders are stateless
        in eval so it makes no numerical difference.
        """
        if do_classifier_free_guidance:
            # An unset negative prompt becomes "" and is genuinely encoded.
            negative_prompt = negative_prompt or ""

        pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
            self.tokenizer, self.text_encoder, prompt
        )
        negative_pooled_prompt_embeds_1 = (
            self._get_clip_prompt_embeds(
                self.tokenizer, self.text_encoder, negative_prompt
            )
            if do_classifier_free_guidance
            else None
        )

        pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
            self.tokenizer_2, self.text_encoder_2, prompt
        )
        negative_pooled_prompt_embeds_2 = (
            self._get_clip_prompt_embeds(
                self.tokenizer_2, self.text_encoder_2, negative_prompt
            )
            if do_classifier_free_guidance
            else None
        )

        # CLIP-L (768) ++ CLIP-G (1280) -> the transformer's 2048-d pooled cond.
        pooled_prompt_embeds = torch.cat(
            [pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1
        )
        negative_pooled_prompt_embeds = (
            torch.cat(
                [negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2],
                dim=-1,
            )
            if do_classifier_free_guidance
            else None
        )

        prompt_embeds_t5 = self._get_t5_prompt_embeds(prompt)
        negative_prompt_embeds_t5 = (
            self._get_t5_prompt_embeds(negative_prompt)
            if do_classifier_free_guidance
            else None
        )

        prompt_embeds_llama3 = self._get_llama3_prompt_embeds(prompt)
        negative_prompt_embeds_llama3 = (
            self._get_llama3_prompt_embeds(negative_prompt)
            if do_classifier_free_guidance
            else None
        )

        return (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = GUIDANCE_SCALE,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = None,
    ):
        """Mirror HiDreamImagePipeline.__call__ and return a PIL image."""
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)

            # Resolution snap, verbatim from __call__: rescale the requested WxH
            # to the model's native pixel budget, then floor both to a multiple
            # of vae_scale_factor * 2. At 1024x1024 this is the identity.
            height, width = self.config.height, self.config.width
            division = self.vae_scale_factor * 2
            s_max = (self.config.default_sample_size * self.vae_scale_factor) ** 2
            scale = math.sqrt(s_max / (width * height))
            width = int(width * scale // division * division)
            height = int(height * scale // division * division)

            # --- 1. Text encoding ---
            logger.info("[STAGE] encode_prompt: 4 text encoders (CPU)")
            (
                prompt_embeds_t5,
                negative_prompt_embeds_t5,
                prompt_embeds_llama3,
                negative_prompt_embeds_llama3,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

            # CFG concat — note the llama3 stream concatenates on dim 1, since
            # dim 0 is the per-layer stack, not the batch.
            if do_classifier_free_guidance:
                prompt_embeds_t5 = torch.cat(
                    [negative_prompt_embeds_t5, prompt_embeds_t5], dim=0
                )
                prompt_embeds_llama3 = torch.cat(
                    [negative_prompt_embeds_llama3, prompt_embeds_llama3], dim=1
                )
                pooled_prompt_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                )

            # --- 2. Latents ---
            # 8x VAE compression, then a further /2 * 2 so the latent dims stay
            # divisible by the 2x2 patch packing (identity at these resolutions).
            num_channels_latents = self.transformer.transformer.config.in_channels
            latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
            latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
            # dtype comes from the pooled embeds (bf16), so the noise is drawn
            # directly in bf16 — matching how the pipeline consumes the generator.
            latents = randn_tensor(
                (batch_size, num_channels_latents, latent_height, latent_width),
                generator=generator,
                device=torch.device(self.device),
                dtype=pooled_prompt_embeds.dtype,
            )

            # --- 3. Timesteps ---
            # UniPC branch of __call__: `mu` / calculate_shift is computed there
            # but only consumed by the FlowMatchEuler path, so it is dropped.
            # __call__ pins the timestep device to cpu whenever torch_xla is
            # importable; on this CPU-only run both branches give "cpu".
            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps

            # --- 4. Denoising loop ---
            for i, t in enumerate(timesteps):
                logger.info("[STEP] transformer step {}/{}", i + 1, num_inference_steps)

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                timestep = t.expand(latent_model_input.shape[0])

                # HiDreamTransformerWrapper == transformer(..., return_dict=False)[0].
                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    prompt_embeds_t5,
                    prompt_embeds_llama3,
                    pooled_prompt_embeds,
                )
                # HiDream predicts the negated flow; the pipeline flips the sign
                # before guidance, not after.
                noise_pred = -noise_pred

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

            # --- 5. VAE decode ---
            logger.info("[STAGE] vae decode (CPU)")
            latents = (
                latents / self.vae.vae.config.scaling_factor
            ) + self.vae.vae.config.shift_factor
            # VAEDecoderWrapper == vae.decode(z, return_dict=False)[0].
            image = self.vae(latents)
            return self.image_processor.postprocess(image, output_type="pil")[0]


# ── Original diffusers pipeline (reference) ────────────────────────────────


def run_original_pipeline():
    # The MoE patch swaps a `.cpu().numpy()` round-trip in moe_infer for the
    # equivalent torch ops so the DiT stays traceable. The manual pipeline gets
    # it for free — ModelLoader applies it when it builds the transformer — so
    # apply it here to put the reference run on the identical code path and make
    # pixel-exact parity a fair bar.
    _patch_hidream_moe_infer()

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_ID)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_ID,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=DTYPE,
    )

    pipe = HiDreamImagePipeline.from_pretrained(
        MODEL_ID,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=DTYPE,
    )
    pipe.to(DEVICE)

    image = pipe(
        PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
    ).images[0]

    # Free ~60 GB before the manual pipeline loads its own copy.
    del pipe, text_encoder_4, tokenizer_4
    gc.collect()
    return image


# ── Our manual pipeline ────────────────────────────────────────────────────


def run_our_pipeline():
    config = HiDreamI1Config(device=DEVICE, dtype=DTYPE)
    pipeline = HiDreamI1Pipeline(config=config)
    pipeline.setup()

    return pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )


# ── Comparison ─────────────────────────────────────────────────────────────


def compare_images(img_a, img_b, name_a="Original", name_b="Ours"):
    arr_a = torch.from_numpy(np.array(img_a))
    arr_b = torch.from_numpy(np.array(img_b))

    logger.info(
        "[PARITY] {} shape: {}   {} shape: {}",
        name_a,
        tuple(arr_a.shape),
        name_b,
        tuple(arr_b.shape),
    )

    identical = arr_a.shape == arr_b.shape and torch.equal(arr_a, arr_b)
    if identical:
        logger.info("[PARITY] RESULT: Pixel-identical!")
    elif arr_a.shape == arr_b.shape:
        # Same CPU, same seed, same dtype — any diff at all is an orchestration
        # bug, but report how far off we are to point at where.
        diff = (arr_a.float() - arr_b.float()).abs()
        total_pixels = arr_a.shape[0] * arr_a.shape[1]
        matching = int((arr_a == arr_b).all(dim=-1).sum())
        logger.info("[PARITY] Max pixel diff : {}", float(diff.max()))
        logger.info("[PARITY] Mean pixel diff: {:.6f}", float(diff.mean()))
        logger.info(
            "[PARITY] Matching pixels: {}/{} ({:.2f}%)",
            matching,
            total_pixels,
            100 * matching / total_pixels,
        )

    assert identical, (
        f"{name_b} is not pixel-identical to {name_a} — the manual pipeline "
        f"does not reproduce HiDreamImagePipeline on CPU"
    )


def test_cpu_parity_check():
    """Stock HiDreamImagePipeline vs the manual ModelLoader-component pipeline."""
    torch.manual_seed(SEED)

    logger.info("--- Running original diffusers HiDreamImagePipeline ---")
    image_org = run_original_pipeline()
    image_org.save("hidream_i1_full_org_n10.png")
    logger.info("Saved hidream_i1_full_org_n10.png")

    logger.info("--- Running our HiDreamI1Pipeline (CPU-only, no TT) ---")
    image_ours = run_our_pipeline()
    image_ours.save("hidream_i1_full_ours_n10.png")
    logger.info("Saved hidream_i1_full_ours_n10.png")

    compare_images(image_org, image_ours)
