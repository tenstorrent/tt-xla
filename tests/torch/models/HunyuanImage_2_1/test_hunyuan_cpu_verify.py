# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanImage 2.1 (Distilled) — CPU-only pipeline parity check.

Compares the stock diffusers ``HunyuanImagePipeline`` (reference) against a
manual pipeline that fetches the 4 nn.Module components via the tt-forge-models
``ModelLoader`` (the same wrappers used by the TT component tests). The manual
pipeline mirrors ``HunyuanImagePipeline.__call__`` on CPU so we can confirm the
component restructuring is correct before doing the e2e TT run — and so a device
switch can later be dropped in around each component call independently.

Distilled-model specifics that shape this pipeline
--------------------------------------------------
* ``model_index.json`` ships no ``guider``/``ocr_guider``, so ``__call__`` falls
  back to ``AdaptiveProjectedMixGuidance(enabled=False)``. With the guider
  disabled ``num_conditions == 1``: the transformer runs ONCE per step on the
  conditional embeds and ``noise_pred`` passes straight through (no CFG concat,
  no negative-prompt encoding). We therefore reimplement the loop as a single
  denoiser call per step — mathematically identical to the guider machinery.
* ``transformer.config.guidance_embeds = True`` → guidance is *distilled*: it is
  fed as an embedding ``guidance = distilled_guidance_scale * 1000`` rather than
  applied as classifier-free rescaling.
* ``transformer.config.use_meanflow = True`` → each step also passes ``timestep_r``
  (the next timestep, or 0 on the final step).
* Text path: Qwen2.5-VL ``.language_model`` penultimate-skip hidden state
  (``hidden_states[-3]``) with the system-template prefix dropped, plus a ByT5
  "glyph" stream for any quoted text in the prompt (this prompt contains
  ``'Tencent'``, so the ByT5 encoder actually runs).
"""

import gc
import re
from typing import Optional

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanImagePipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from loguru import logger
from PIL import Image
from transformers import ByT5Tokenizer, Qwen2Tokenizer

from third_party.tt_forge_models.hunyuan_image_2_1.pytorch import (
    ModelLoader,
    ModelVariant,
)


REPO_ID = "hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers"
# Canonical HunyuanImage-2.1 distilled sample prompt (from test_hunyuan_cpu.py /
# the HF model card). The single-quoted 'Tencent' triggers the ByT5 glyph path.
PROMPT = (
    "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, "
    "standing in a painting studio, wearing a red knitted scarf and a red beret "
    "with the word 'Tencent' on it, holding a paintbrush with a focused "
    "expression as it paints an oil painting of the Mona Lisa, rendered in a "
    "photorealistic photographic style."
)
# Guider is disabled for the distilled model, so negative_prompt is never used.
NEGATIVE_PROMPT = None
SEED = 649151
NUM_INFERENCE_STEPS = 8
DISTILLED_GUIDANCE_SCALE = 3.5
HEIGHT = 2048
WIDTH = 2048
DEVICE = "cpu"

# Qwen2.5-VL prompt template + the number of leading template tokens to drop.
# Verbatim from HunyuanImagePipeline.__init__.
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34
TOKENIZER_MAX_LENGTH = 1000  # Qwen2.5-VL, before the template-prefix drop
TOKENIZER_2_MAX_LENGTH = 128  # ByT5
# hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)] == hidden_states[-3]
HIDDEN_STATE_SKIP_LAYER = 2


# Copied verbatim from diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage
def extract_glyph_text(prompt: str):
    """Extract quoted text for glyph (ByT5) rendering, or None if none found."""
    text_prompt_texts = []
    pattern_quote_single = r"\'(.*?)\'"
    pattern_quote_double = r"\"(.*?)\""
    pattern_quote_chinese_single = r"‘(.*?)’"
    pattern_quote_chinese_double = r"“(.*?)”"

    text_prompt_texts.extend(re.findall(pattern_quote_single, prompt))
    text_prompt_texts.extend(re.findall(pattern_quote_double, prompt))
    text_prompt_texts.extend(re.findall(pattern_quote_chinese_single, prompt))
    text_prompt_texts.extend(re.findall(pattern_quote_chinese_double, prompt))

    if text_prompt_texts:
        return ". ".join([f'Text "{text}"' for text in text_prompt_texts]) + ". "
    return None


# ── Manual pipeline (CPU mirror of HunyuanImagePipeline.__call__) ──────────


class HunyuanImageConfig:
    def __init__(self, device="cpu"):
        self.repo_id = REPO_ID
        self.height = HEIGHT
        self.width = WIDTH
        self.tokenizer_max_length = TOKENIZER_MAX_LENGTH
        self.tokenizer_2_max_length = TOKENIZER_2_MAX_LENGTH
        self.device = device


class HunyuanImagePipelineManual:
    """Manual HunyuanImage pipeline using nn.Module components from tt-forge-models."""

    def __init__(self, config: HunyuanImageConfig):
        self.config = config
        self.device = config.device
        self.repo_id = config.repo_id

    def setup(self):
        logger.info("[setup] loading components / scheduler / tokenizers ...")
        self.load_models()
        logger.info("[setup] loading scheduler ...")
        self.load_scheduler()
        logger.info("[setup] loading tokenizers ...")
        self.load_tokenizers()
        # Read model-driven constants off the loaded components (avoids magic
        # numbers and keeps us pinned to whatever the checkpoint actually says).
        self.vae_scale_factor = self.vae.vae.config.spatial_compression_ratio
        self.scaling_factor = self.vae.vae.config.scaling_factor
        self.num_channels_latents = self.transformer.transformer.config.in_channels
        self.text_encoder_2_dim = self.text_encoder_2.config.d_model
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        logger.info(
            "[setup] done: vae_scale_factor={}, scaling_factor={}, "
            "num_channels_latents={}, text_encoder_2_dim={}",
            self.vae_scale_factor,
            self.scaling_factor,
            self.num_channels_latents,
            self.text_encoder_2_dim,
        )

    def load_models(self):
        # TEXT_ENCODER   → raw Qwen2.5-VL .language_model (call w/ output_hidden_states)
        # TEXT_ENCODER_2 → raw ByT5 T5EncoderModel
        # TRANSFORMER    → HunyuanImage21TransformerWrapper (tensor-in/tensor-out)
        # VAE            → VAEDecoderWrapper (decode-only)
        logger.info("[load_models] text_encoder (Qwen2.5-VL .language_model, ~8.29B) ...")
        self.text_encoder = ModelLoader(ModelVariant.TEXT_ENCODER).load_model(
            dtype_override=torch.float32
        )
        logger.info("[load_models] text_encoder_2 (ByT5, ~0.22B) ...")
        self.text_encoder_2 = ModelLoader(ModelVariant.TEXT_ENCODER_2).load_model(
            dtype_override=torch.float32
        )
        logger.info("[load_models] transformer (HunyuanImage MMDiT, ~17.45B) ...")
        self.transformer = ModelLoader(ModelVariant.TRANSFORMER).load_model(
            dtype_override=torch.float32
        )
        logger.info("[load_models] vae (AutoencoderKLHunyuanImage decoder, ~0.41B) ...")
        self.vae = ModelLoader(ModelVariant.VAE).load_model(
            dtype_override=torch.float32
        )
        logger.info("[load_models] all components loaded (dtype=float32)")

    def load_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.repo_id, subfolder="scheduler"
        )

    def load_tokenizers(self):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            self.repo_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
            self.repo_id, subfolder="tokenizer_2"
        )

    def _get_qwen_prompt_embeds(self, prompt):
        """Qwen2.5-VL → (prompt_embeds (1,1000,3584), mask (1,1000))."""
        drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
        txt = [PROMPT_TEMPLATE_ENCODE.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            max_length=self.config.tokenizer_max_length + drop_idx,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # tt-forge-models returns the raw .language_model; call it exactly as the
        # pipeline calls the full text_encoder (text-only path → same hidden states).
        logger.info(
            "[qwen] running Qwen2.5-VL text encoder forward (input_ids={}) — "
            "slow on CPU fp32 ...",
            tuple(txt_tokens.input_ids.shape),
        )
        outputs = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        logger.info("[qwen] text encoder forward done")
        prompt_embeds = outputs.hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]

        # Drop the fixed system-template prefix.
        prompt_embeds = prompt_embeds[:, drop_idx:]
        encoder_attention_mask = txt_tokens.attention_mask[:, drop_idx:]

        prompt_embeds = prompt_embeds.to(dtype=torch.float32, device=self.device)
        encoder_attention_mask = encoder_attention_mask.to(device=self.device)
        return prompt_embeds, encoder_attention_mask

    def _get_byt5_prompt_embeds(self, glyph_text):
        """ByT5 → (glyph_embeds (1,128,1472), mask (1,128))."""
        txt_tokens = self.tokenizer_2(
            glyph_text,
            padding="max_length",
            max_length=self.config.tokenizer_2_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        logger.info("[byt5] running ByT5 glyph encoder forward (glyph={!r}) ...", glyph_text)
        prompt_embeds = self.text_encoder_2(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask.float(),
        )[0]
        logger.info("[byt5] glyph encoder forward done")

        prompt_embeds = prompt_embeds.to(dtype=torch.float32, device=self.device)
        # Keep the integer mask (not the float one fed to the encoder).
        encoder_attention_mask = txt_tokens.attention_mask.to(device=self.device)
        return prompt_embeds, encoder_attention_mask

    def _encode_prompt(self, prompt: str):
        """Run both text encoders → qwen (embeds, mask) + byt5 glyph (embeds, mask)."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt)

        prompt_embeds_2_list = []
        prompt_embeds_mask_2_list = []
        for glyph_text in [extract_glyph_text(p) for p in prompt]:
            if glyph_text is None:
                # No quoted text → zero glyph stream (matches pipeline).
                glyph_embeds = torch.zeros(
                    (1, self.config.tokenizer_2_max_length, self.text_encoder_2_dim),
                    device=self.device,
                )
                glyph_mask = torch.zeros(
                    (1, self.config.tokenizer_2_max_length),
                    device=self.device,
                    dtype=torch.int64,
                )
            else:
                glyph_embeds, glyph_mask = self._get_byt5_prompt_embeds(glyph_text)
            prompt_embeds_2_list.append(glyph_embeds)
            prompt_embeds_mask_2_list.append(glyph_mask)

        prompt_embeds_2 = torch.cat(prompt_embeds_2_list, dim=0)
        prompt_embeds_mask_2 = torch.cat(prompt_embeds_mask_2_list, dim=0)

        # num_images_per_prompt == 1 → the pipeline's repeat/view reshaping is a
        # no-op, so the shapes above are already what the transformer expects.
        logger.info(
            "[encode] prompt_embeds={}, mask={}, glyph_embeds={}, glyph_mask={}",
            tuple(prompt_embeds.shape),
            tuple(prompt_embeds_mask.shape),
            tuple(prompt_embeds_2.shape),
            tuple(prompt_embeds_mask_2.shape),
        )
        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,  # unused: guider disabled (distilled)
        distilled_guidance_scale: float = 3.5,
        num_inference_steps: int = 8,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = 1
        device = self.device

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # --- Text encoding ---
            logger.info("[generate] encoding prompt ...")
            (
                prompt_embeds,
                prompt_embeds_mask,
                prompt_embeds_2,
                prompt_embeds_mask_2,
            ) = self._encode_prompt(prompt)

            # --- Latents (transformer.config.in_channels channels) ---
            latents_height = int(self.config.height) // self.vae_scale_factor
            latents_width = int(self.config.width) // self.vae_scale_factor
            latent_shape = (
                batch_size,
                self.num_channels_latents,
                latents_height,
                latents_width,
            )
            latents = randn_tensor(
                latent_shape, generator=generator, device=device, dtype=torch.float32
            )
            logger.info("[generate] latents={}", tuple(latents.shape))

            # --- Timesteps (flow-matching, custom linear sigmas) ---
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
            timesteps = self.scheduler.timesteps
            logger.info("[generate] {} timesteps ready", len(timesteps))

            # --- Distilled guidance embedding (guidance_embeds=True) ---
            guidance = (
                torch.tensor(
                    [distilled_guidance_scale] * latents.shape[0],
                    dtype=torch.float32,
                    device=device,
                )
                * 1000.0
            )

            # --- Denoising loop ---
            self.scheduler.set_begin_index(0)
            for i, t in enumerate(timesteps):
                logger.info(
                    "[generate] denoising step {}/{} (t={:.4f}) — transformer forward ...",
                    i + 1,
                    num_inference_steps,
                    float(t),
                )

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # use_meanflow=True → refiner timestep = next timestep (0 on last).
                if i == len(timesteps) - 1:
                    timestep_r = torch.tensor([0.0], device=device)
                else:
                    timestep_r = timesteps[i + 1]
                timestep_r = timestep_r.expand(latents.shape[0]).to(latents.dtype)

                # Single conditional forward (guider disabled → no CFG). The
                # wrapper signature is positional/tensor-only.
                noise_pred = self.transformer(
                    latents,
                    timestep,
                    timestep_r,
                    guidance,
                    prompt_embeds,
                    prompt_embeds_mask,
                    prompt_embeds_2,
                    prompt_embeds_mask_2,
                )

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

            # --- VAE decode ---
            logger.info("[generate] VAE decode ...")
            latents = latents.to(torch.float32) / self.scaling_factor
            image = self.vae(latents)  # VAEDecoderWrapper.decode → (1, 3, H, W)
            logger.info("[generate] VAE decode done, image={}", tuple(image.shape))
            return image


# ── Original diffusers pipeline (reference) ────────────────────────────────


def run_original_pipeline():
    logger.info("[original] loading stock HunyuanImagePipeline ...")
    pipe = HunyuanImagePipeline.from_pretrained(REPO_ID, torch_dtype=torch.float32)
    pipe.to(DEVICE)

    logger.info("[original] running {} denoising steps ...", NUM_INFERENCE_STEPS)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    image = pipe(
        PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        distilled_guidance_scale=DISTILLED_GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    ).images[0]

    # Free the ~100GB of fp32 weights before loading our components.
    logger.info("[original] done; freeing stock pipeline weights ...")
    del pipe
    gc.collect()
    return image


# ── Our manual pipeline ────────────────────────────────────────────────────


def run_our_pipeline():
    config = HunyuanImageConfig(device=DEVICE)
    pipeline = HunyuanImagePipelineManual(config=config)
    pipeline.setup()

    img_tensor = pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        distilled_guidance_scale=DISTILLED_GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    # Same post-processing the stock pipeline applies (do_normalize=True).
    image = pipeline.image_processor.postprocess(img_tensor, output_type="pil")[0]

    # Free the ~100GB of fp32 component weights before the stock pipeline loads
    # (they run back-to-back; keeping both resident would ~double peak RAM).
    logger.info("[ours] done; freeing manual pipeline component weights ...")
    del img_tensor, pipeline
    gc.collect()
    return image


# ── Comparison ─────────────────────────────────────────────────────────────


def compare_images(img_a, img_b, name_a="Original", name_b="Ours"):
    arr_a = np.array(img_a)
    arr_b = np.array(img_b)

    logger.info("{}", "=" * 60)
    logger.info("  {} shape: {}   {} shape: {}", name_a, arr_a.shape, name_b, arr_b.shape)

    if np.array_equal(arr_a, arr_b):
        logger.info("  RESULT: Pixel-identical!")
    else:
        diff = np.abs(arr_a.astype(np.float32) - arr_b.astype(np.float32))
        total_pixels = arr_a.shape[0] * arr_a.shape[1]
        matching = (arr_a == arr_b).all(axis=-1).sum()
        logger.info("  Max pixel diff : {}", diff.max())
        logger.info("  Mean pixel diff: {:.6f}", diff.mean())
        logger.info(
            "  Matching pixels: {}/{} ({:.2f}%)",
            matching,
            total_pixels,
            100 * matching / total_pixels,
        )
    logger.info("{}", "=" * 60)


def test_cpu_check():
    # Manual pipeline first: it's the code under test, so fail fast (~22 min)
    # instead of paying for the reference run before hitting any issue. Each
    # pipeline is freed before the next loads, so order does not change results
    # (independent seeded generators) or peak memory.
    logger.info("--- Running our HunyuanImagePipelineManual (CPU-only, no TT) ---")
    image_ours = run_our_pipeline()
    image_ours.save("hyimage_ours.png")
    logger.info("Saved hyimage_ours.png")

    logger.info("--- Running original diffusers pipeline ---")
    image_org = run_original_pipeline()
    image_org.save("hyimage_org.png")
    logger.info("Saved hyimage_org.png")

    compare_images(image_org, image_ours)


if __name__ == "__main__":
    test_cpu_check()
