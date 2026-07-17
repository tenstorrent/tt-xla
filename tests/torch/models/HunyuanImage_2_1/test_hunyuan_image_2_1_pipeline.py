# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanImage 2.1 (Distilled) — nightly e2e pipeline test with per-component PCC.

Every component runs on Tenstorrent (qb2 Blackhole); after each TT forward the same
tensors are run through a fp32 CPU twin and PCC is asserted, failing fast below
``PCC_THRESHOLD``. Only the transformer is tensor-parallel sharded (it does not fit
on a single chip); the text encoders and VAE run on a single chip.
"""

import gc
import re
from typing import Optional

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from transformers import ByT5Tokenizer, Qwen2Tokenizer
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.hunyuan_image_2_1.pytorch import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.hunyuan_image_2_1.pytorch.src.model_utils import (
    NUM_CHANNELS_LATENTS,
    VAE_SCALE_FACTOR,
)

REPO_ID = "hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers"
PROMPT = (
    "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, "
    "standing in a painting studio, wearing a red knitted scarf and a red beret "
    "with the word 'Tencent' on it, holding a paintbrush with a focused "
    "expression as it paints an oil painting of the Mona Lisa, rendered in a "
    "photorealistic photographic style."
)
SEED = 649151
NUM_INFERENCE_STEPS = 8
DISTILLED_GUIDANCE_SCALE = 3.5
HEIGHT = 2048
WIDTH = 2048
PCC_THRESHOLD = 0.80

# Verbatim from HunyuanImagePipeline.__init__.
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34
TOKENIZER_MAX_LENGTH = 1000  # Qwen2.5-VL, before the template-prefix drop
TOKENIZER_2_MAX_LENGTH = 128  # ByT5
HIDDEN_STATE_SKIP_LAYER = 2  # hidden_states[-(SKIP + 1)] == hidden_states[-3]


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _log_io(component: str, **tensors):
    """Log shape + dtype of each component input (to compare against loader randn)."""
    for name, t in tensors.items():
        logger.info(
            "[io] {}: {} shape={} dtype={}", component, name, tuple(t.shape), t.dtype
        )


# Copied verbatim from diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage
def extract_glyph_text(prompt: str):
    """Extract quoted text for glyph (ByT5) rendering, or None if none found."""
    text_prompt_texts = []
    text_prompt_texts.extend(re.findall(r"\'(.*?)\'", prompt))
    text_prompt_texts.extend(re.findall(r"\"(.*?)\"", prompt))
    text_prompt_texts.extend(re.findall(r"‘(.*?)’", prompt))
    text_prompt_texts.extend(re.findall(r"“(.*?)”", prompt))
    if text_prompt_texts:
        return ". ".join([f'Text "{text}"' for text in text_prompt_texts]) + ". "
    return None


class HunyuanImageConfig:
    def __init__(self, device: str = "cpu"):
        self.repo_id = REPO_ID
        self.height = HEIGHT
        self.width = WIDTH
        self.tokenizer_max_length = TOKENIZER_MAX_LENGTH
        self.tokenizer_2_max_length = TOKENIZER_2_MAX_LENGTH
        self.device = device


class HunyuanImagePipeline:
    """HunyuanImage 2.1 pipeline: all four components on TT with per-component PCC."""

    def __init__(self, config: HunyuanImageConfig):
        self.config = config
        self.device = config.device
        self.repo_id = config.repo_id

    def setup(self):
        # SPMD mesh: only the transformer is sharded across it; other modules
        # run replicated (single chip).
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = ModelLoader(ModelVariant.TRANSFORMER).get_mesh_config(
            self.num_devices
        )
        self.mesh = get_mesh(mesh_shape, mesh_names)
        logger.info("[setup] mesh {} over {} device(s)", mesh_shape, self.num_devices)
        self.load_scheduler()
        self.load_tokenizers()
        self._cpu_twins = {}

    def _cpu_twin(self, variant: ModelVariant, dtype=torch.float32):
        # Lazy CPU reference for PCC, one per (first-seen) variant.
        if variant not in self._cpu_twins:
            logger.info("[PCC] loading CPU twin: {} ({})", variant, dtype)
            self._cpu_twins[variant] = ModelLoader(variant).load_model(
                dtype_override=dtype
            )
        return self._cpu_twins[variant]

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

    def _encode_qwen(self, prompt):
        """Qwen2.5-VL text encoder — run on CPU (fp32); TT PCC is too low for now."""
        logger.info("[STAGE] text_encoder (Qwen): CPU (not on TT)")
        drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
        tokens = self.tokenizer(
            [PROMPT_TEMPLATE_ENCODE.format(prompt)],
            max_length=self.config.tokenizer_max_length + drop_idx,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = tokens.input_ids, tokens.attention_mask
        skip = HIDDEN_STATE_SKIP_LAYER
        _log_io("text_encoder", input_ids=input_ids, attention_mask=attention_mask)

        # Penultimate-skip hidden state, template prefix dropped (diffusers
        # _get_qwen_prompt_embeds).
        out = self._cpu_twin(ModelVariant.TEXT_ENCODER)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = out.hidden_states[-(skip + 1)][:, drop_idx:].to(torch.float32)
        prompt_embeds_mask = attention_mask[:, drop_idx:]
        return prompt_embeds, prompt_embeds_mask

    def _encode_byt5(self, prompt, dev):
        """ByT5 glyph encoder (single chip) → (glyph_embeds, mask)."""
        logger.info("[STAGE] text_encoder_2 (ByT5): start")
        glyph_text = extract_glyph_text(prompt)

        if glyph_text is None:
            # No quoted text → zero glyph stream (matches pipeline); no TT run.
            dim = self._cpu_twin(ModelVariant.TEXT_ENCODER_2).config.d_model
            embeds = torch.zeros((1, self.config.tokenizer_2_max_length, dim))
            mask = torch.zeros(
                (1, self.config.tokenizer_2_max_length), dtype=torch.int64
            )
            return embeds, mask

        tokens = self.tokenizer_2(
            glyph_text,
            padding="max_length",
            max_length=self.config.tokenizer_2_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids
        attention_mask_f = tokens.attention_mask.float()  # encoder takes float mask
        prompt_embeds_mask_2 = tokens.attention_mask  # int mask flows downstream
        _log_io("text_encoder_2", input_ids=input_ids, attention_mask=attention_mask_f)

        # Pipeline uses the last_hidden_state (out[0]).
        golden = self._cpu_twin(ModelVariant.TEXT_ENCODER_2)(
            input_ids=input_ids, attention_mask=attention_mask_f
        )[0]

        encoder = ModelLoader(ModelVariant.TEXT_ENCODER_2).load_model(
            dtype_override=torch.float32
        )
        encoder = encoder.to(dev)
        compiled = torch.compile(encoder, backend="tt")
        out = compiled(
            input_ids=input_ids.to(dev), attention_mask=attention_mask_f.to(dev)
        )
        prompt_embeds_2 = out[0].cpu().to(torch.float32)

        encoder = encoder.to("cpu")
        del compiled, encoder
        gc.collect()
        torch_xla.sync()

        pcc = _pcc(prompt_embeds_2, golden)
        logger.info("[PCC] text_encoder_2 (ByT5): pcc={:.6f}", pcc)
        assert (
            pcc >= PCC_THRESHOLD
        ), f"text_encoder_2 PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"
        return prompt_embeds_2, prompt_embeds_mask_2

    def generate(
        self,
        prompt: str,
        distilled_guidance_scale: float = DISTILLED_GUIDANCE_SCALE,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = 1
        dev = torch_xla.device()

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)

            prompt_embeds, prompt_embeds_mask = self._encode_qwen(prompt)
            prompt_embeds_2, prompt_embeds_mask_2 = self._encode_byt5(prompt, dev)

            # Switch to optimization_level=1 after the encoders (sticky): required
            # by the transformer, and applied to the VAE too. ByT5 ran at default.
            torch_xla.set_custom_compile_options({"optimization_level": 1})

            # Latents: fp32 trajectory, advanced by the TT outputs.
            latents_h = int(self.config.height) // VAE_SCALE_FACTOR
            latents_w = int(self.config.width) // VAE_SCALE_FACTOR
            latents = randn_tensor(
                (batch_size, NUM_CHANNELS_LATENTS, latents_h, latents_w),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )

            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            self.scheduler.set_timesteps(sigmas=sigmas, device="cpu")
            timesteps = self.scheduler.timesteps
            self.scheduler.set_begin_index(0)

            # Distilled guidance embedding (guidance_embeds=True).
            guidance = (
                torch.tensor(
                    [distilled_guidance_scale] * batch_size, dtype=torch.float32
                )
                * 1000.0
            )

            # Transformer: sharded, bf16 on TT; placed once for the whole loop.
            logger.info(
                "[STAGE] transformer (sharded, bf16): start ({} steps)",
                num_inference_steps,
            )
            tr_loader = ModelLoader(ModelVariant.TRANSFORMER)
            transformer = tr_loader.load_model(dtype_override=torch.bfloat16).to(dev)
            specs = tr_loader.load_shard_spec(transformer)
            assert specs, "transformer shard spec is empty — would run replicated/OOM"
            for tensor, spec in specs.items():
                xs.mark_sharding(tensor, self.mesh, spec)
            tt_transformer = torch.compile(transformer, backend="tt")
            twin_transformer = self._cpu_twin(
                ModelVariant.TRANSFORMER, dtype=torch.bfloat16
            )

            tt_bf16 = lambda x: x.to(torch.bfloat16).to(dev)
            tt_int = lambda x: x.to(dev)  # masks stay integer
            cpu_bf16 = lambda x: x.to(torch.bfloat16)  # bf16 CPU golden inputs

            for i, t in enumerate(timesteps):
                logger.info("[STEP] transformer step {}/{}", i + 1, num_inference_steps)
                timestep = t.expand(batch_size).to(latents.dtype)
                # meanflow: refiner timestep = next timestep (0 on the last step).
                if i == len(timesteps) - 1:
                    timestep_r = torch.tensor([0.0])
                else:
                    timestep_r = timesteps[i + 1]
                timestep_r = timestep_r.expand(batch_size).to(latents.dtype)

                # Single conditional forward (distilled guider disabled → no CFG).
                tt_inputs = [
                    tt_bf16(latents),
                    tt_bf16(timestep),
                    tt_bf16(timestep_r),
                    tt_bf16(guidance),
                    tt_bf16(prompt_embeds),
                    tt_int(prompt_embeds_mask),
                    tt_bf16(prompt_embeds_2),
                    tt_int(prompt_embeds_mask_2),
                ]
                if i == 0:
                    _log_io(
                        "transformer",
                        hidden_states=tt_inputs[0],
                        timestep=tt_inputs[1],
                        timestep_r=tt_inputs[2],
                        guidance=tt_inputs[3],
                        encoder_hidden_states=tt_inputs[4],
                        encoder_attention_mask=tt_inputs[5],
                        encoder_hidden_states_2=tt_inputs[6],
                        encoder_attention_mask_2=tt_inputs[7],
                    )
                noise_pred = tt_transformer(*tt_inputs).cpu().to(torch.float32)

                # CPU golden in bf16 (matches the TT dtype).
                golden_noise = twin_transformer(
                    cpu_bf16(latents),
                    cpu_bf16(timestep),
                    cpu_bf16(timestep_r),
                    cpu_bf16(guidance),
                    cpu_bf16(prompt_embeds),
                    prompt_embeds_mask,
                    cpu_bf16(prompt_embeds_2),
                    prompt_embeds_mask_2,
                ).to(torch.float32)
                pcc = _pcc(noise_pred, golden_noise)
                logger.info(
                    "[PCC] transformer step {}/{}: pcc={:.6f}",
                    i + 1,
                    num_inference_steps,
                    pcc,
                )
                assert pcc >= PCC_THRESHOLD, (
                    f"transformer step {i + 1}/{num_inference_steps} PCC {pcc:.6f} "
                    f"below threshold {PCC_THRESHOLD}"
                )

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

            transformer = transformer.to("cpu")
            del tt_transformer, transformer
            gc.collect()
            torch_xla.sync()
            logger.info("[STAGE] transformer: done")

            # VAE decode (single chip, fp32).
            logger.info("[STAGE] vae: start")
            vae_twin = self._cpu_twin(ModelVariant.VAE)
            latents = latents.to(torch.float32) / vae_twin.vae.config.scaling_factor

            vae = ModelLoader(ModelVariant.VAE).load_model(
                dtype_override=torch.float32
            )
            vae = vae.to(dev)
            tt_vae = torch.compile(vae, backend="tt")

            _log_io("vae", z=latents)
            image = tt_vae(latents.to(dev)).cpu().to(torch.float32)
            golden_image = vae_twin(latents)

            vae = vae.to("cpu")
            del tt_vae, vae
            gc.collect()
            torch_xla.sync()

            pcc = _pcc(image, golden_image)
            logger.info("[PCC] vae: pcc={:.6f}", pcc)
            assert (
                pcc >= PCC_THRESHOLD
            ), f"vae PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"
            logger.info("[STAGE] vae: done")
            return image


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.qb2_blackhole
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="HunyuanImage21_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_hunyuan_image_2_1_pipeline(monkeypatch):
    """HunyuanImage 2.1 pipeline (all components on TT) with per-component PCC."""
    # Transformer attention consumes a bool mask; opt in to catching the SDPA
    # composite for bool masks (auto-restored after the test).
    monkeypatch.setenv("TT_XLA_CATCH_BOOL_MASK_SDPA", "1")
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    config = HunyuanImageConfig(device="cpu")
    pipeline = HunyuanImagePipeline(config=config)
    pipeline.setup()
    pipeline.generate(
        prompt=PROMPT,
        distilled_guidance_scale=DISTILLED_GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
