# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HiDream-I1-Full — nightly transformer PCC test on Tenstorrent.

Only the transformer (Sparse-MoE MM-DiT, tensor-parallel sharded) runs on TT;
the CLIP/T5/Llama text encoders, the scheduler and the VAE all run on CPU. Every
component is bfloat16, the dtype the model ships in. Each denoising step runs the
transformer on TT and compares its noise prediction against a CPU twin fed the
same inputs, asserting PCC >= ``PCC_THRESHOLD``.
"""

import gc
import math
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from tt_torch.sparse_mlp import enable_sparse_mlp
from transformers import CLIPTokenizer, PreTrainedTokenizerFast, T5Tokenizer
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.hidream_i1.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.hidream_i1.pytorch.src.model_utils import (
    HIDREAM_REPO_ID,
    LATENT_CHANNELS,
    LLAMA_REPO_ID,
    MAX_SEQ_LEN,
    VAE_SCALE,
)

REPO_ID = HIDREAM_REPO_ID
LLAMA_ID = LLAMA_REPO_ID
PROMPT = 'A cat holding a sign that says "HiDream.ai".'
# encode_prompt turns an unset negative prompt into "" and encodes it — there is
# no force_zeros_for_empty_prompt shortcut.
NEGATIVE_PROMPT = None
SEED = 0
NUM_INFERENCE_STEPS = 10 # smoke run
# Per the model card: Full -> 5.0 / 50. > 1 enables CFG, doubling the DiT batch.
GUIDANCE_SCALE = 5.0
HEIGHT = 1024
WIDTH = 1024
DEFAULT_SAMPLE_SIZE = 128  # HiDreamImagePipeline.default_sample_size
DTYPE = torch.bfloat16
PCC_THRESHOLD = 0.90

# Stand-in config for enable_sparse_mlp: create_a2a_from_deepseek_v3_moe looks up
# DeepSeek's attribute names, and HiDream's config spells it
# num_activated_experts -- without num_experts_per_tok it falls back to its
# default of 6.
MOE_CONFIG = SimpleNamespace(n_routed_experts=4, num_experts_per_tok=2)


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _strip_cpu_golden(ff) -> None:
    """Drop the pre-stack expert Linears `enable_sparse_mlp` keeps for its own
    golden-eval fallback. They are unsharded, so `.to(dev)` would replicate
    ~20 GB of unused weights onto every device; the PCC golden here is a
    separate CPU twin.
    """
    mlp = getattr(ff, "mlp", None)
    if mlp is None:
        return
    if hasattr(mlp, "_original_mlp"):
        object.__setattr__(mlp, "_original_mlp", None)
    experts = getattr(mlp, "experts", None)
    if experts is not None and "original_experts" in getattr(experts, "_modules", {}):
        del experts._modules["original_experts"]


def _pcc(device_out, golden_out) -> float:
    # Upcast for the metric only: the PCC reduction accumulates in the input
    # dtype, and bf16 accumulation over ~500 k elements would swamp the signal.
    return float(
        _PCC_EVALUATOR._compare_pcc(
            device_out.float(), golden_out.float(), _PCC_CONFIG
        )
    )


class HiDreamI1Config:
    def __init__(self, device: str = "cpu"):
        self.repo_id = REPO_ID
        self.llama_id = LLAMA_ID
        self.height = HEIGHT
        self.width = WIDTH
        self.max_sequence_length = MAX_SEQ_LEN
        self.default_sample_size = DEFAULT_SAMPLE_SIZE
        self.vae_scale_factor = VAE_SCALE
        self.device = device


class HiDreamI1Pipeline:
    """Transformer on TT (bf16, sharded) with per-step PCC; encoders + VAE on CPU."""

    def __init__(self, config: HiDreamI1Config):
        self.config = config
        self.device = config.device
        self.repo_id = config.repo_id

    def setup(self):
        # SPMD mesh for the sharded transformer — the only module that runs on TT.
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = ModelLoader(ModelVariant.TRANSFORMER).get_mesh_config(
            self.num_devices
        )
        self.mesh = get_mesh(mesh_shape, mesh_names)
        self.mesh_shape = mesh_shape
        logger.info("[setup] mesh {} over {} device(s)", mesh_shape, self.num_devices)
        self.load_scheduler()
        self.load_tokenizers()
        self._cpu_twins = {}

    def _cpu_twin(self, variant: ModelVariant, dtype=DTYPE):
        # Lazy CPU model, one per (first-seen) variant. Used for the CPU-only
        # components and as the golden for the transformer PCC.
        if variant not in self._cpu_twins:
            logger.info("[load] CPU model: {} ({})", variant, dtype)
            self._cpu_twins[variant] = ModelLoader(variant).load_model(
                dtype_override=dtype
            )
        return self._cpu_twins[variant]

    def load_scheduler(self):
        # model_index.json pins UniPCMultistepScheduler for HiDream-I1-Full.
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.repo_id, subfolder="scheduler"
        )

    def load_tokenizers(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.repo_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.repo_id, subfolder="tokenizer_2"
        )
        self.tokenizer_3 = T5Tokenizer.from_pretrained(
            self.repo_id, subfolder="tokenizer_3"
        )
        # Not in the HiDream snapshot; the pipeline expects the caller to supply it.
        self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(self.config.llama_id)
        # HiDreamImagePipeline.__init__ does this; Llama ships no pad token and
        # padding="max_length" would raise without it.
        self.tokenizer_4.pad_token = self.tokenizer_4.eos_token

    def _get_clip_prompt_embeds(self, tokenizer, variant: ModelVariant, prompt: str):
        """CLIP-L / CLIP-G pooled embedding — CPU (bf16)."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=min(self.config.max_sequence_length, 218),  # 218, not 77
            truncation=True,
            return_tensors="pt",
        )
        # CLIPPooledWrapper == text_encoder(input_ids, output_hidden_states=True)[0].
        return self._cpu_twin(variant)(text_inputs.input_ids)

    def _get_t5_prompt_embeds(self, prompt: str):
        """T5-XXL encoder — CPU (bf16)."""
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
        return self._cpu_twin(ModelVariant.TEXT_ENCODER_3)(
            text_inputs.input_ids, text_inputs.attention_mask
        )

    def _get_llama3_prompt_embeds(self, prompt: str):
        """Llama-3.1-8B encoder — CPU (bf16)."""
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
        # LlamaStackedHiddenWrapper == stack(hidden_states[1:], dim=0) -> (32,1,128,4096).
        return self._cpu_twin(ModelVariant.TEXT_ENCODER_4)(
            text_inputs.input_ids, text_inputs.attention_mask
        )

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        do_classifier_free_guidance: bool = True,
    ):
        """Mirror HiDreamImagePipeline.encode_prompt at batch_size=1, 1 image/prompt."""
        logger.info("[STAGE] text encoders (CLIP-L, CLIP-G, T5, Llama): CPU (not on TT)")
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""

        pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
            self.tokenizer, ModelVariant.TEXT_ENCODER, prompt
        )
        negative_pooled_prompt_embeds_1 = (
            self._get_clip_prompt_embeds(
                self.tokenizer, ModelVariant.TEXT_ENCODER, negative_prompt
            )
            if do_classifier_free_guidance
            else None
        )

        pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
            self.tokenizer_2, ModelVariant.TEXT_ENCODER_2, prompt
        )
        negative_pooled_prompt_embeds_2 = (
            self._get_clip_prompt_embeds(
                self.tokenizer_2, ModelVariant.TEXT_ENCODER_2, negative_prompt
            )
            if do_classifier_free_guidance
            else None
        )

        # CLIP-L (768) ++ CLIP-G (1280) -> the DiT's 2048-d pooled conditioning.
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
    ) -> torch.Tensor:
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1
        dev = torch_xla.device()

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)

            # Resolution snap from __call__: rescale to the model's pixel budget,
            # then floor to a multiple of vae_scale_factor * 2. Identity at 1024.
            height, width = self.config.height, self.config.width
            division = self.config.vae_scale_factor * 2
            s_max = (self.config.default_sample_size * self.config.vae_scale_factor) ** 2
            scale = math.sqrt(s_max / (width * height))
            width = int(width * scale // division * division)
            height = int(height * scale // division * division)

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

            # CFG concat — this is what makes the DiT batch 2. llama3 concatenates
            # on dim 1, since dim 0 is the 32-layer stack, not the batch.
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

            # Latents: bf16 trajectory (dtype follows the pooled embeds, as in the
            # stock pipeline), advanced by the TT transformer outputs.
            latents_h = 2 * (int(height) // (self.config.vae_scale_factor * 2))
            latents_w = 2 * (int(width) // (self.config.vae_scale_factor * 2))
            latents = randn_tensor(
                (batch_size, LATENT_CHANNELS, latents_h, latents_w),
                generator=generator,
                device=torch.device("cpu"),
                dtype=pooled_prompt_embeds.dtype,
            )

            # UniPC branch of __call__; `mu` is only used by the FlowMatchEuler path.
            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps

            # Transformer: sharded, bf16 on TT; placed once for the whole loop.
            logger.info(
                "[STAGE] transformer (sharded, bf16): start ({} steps)",
                num_inference_steps,
            )
            tr_loader = ModelLoader(ModelVariant.TRANSFORMER)
            transformer = tr_loader.load_model(dtype_override=DTYPE)
            # Swap the MoE blocks before .to(dev): the swap stacks the
            # per-expert Linears into new parameters.
            # cluster_axis=1: mesh is (1, 4), so axis 0 would dispatch nowhere.
            transformer = enable_sparse_mlp(
                transformer,
                mesh=self.mesh_shape,
                cluster_axis=1,
                config=MOE_CONFIG,
            )
            for module in transformer.modules():
                _strip_cpu_golden(module)
            transformer = transformer.to(dev)
            specs = tr_loader.load_shard_spec(transformer)
            assert specs, "transformer shard spec is empty — would run replicated/OOM"
            for tensor, spec in specs.items():
                xs.mark_sharding(tensor, self.mesh, spec)
            tt_transformer = torch.compile(transformer, backend="tt")
            twin_transformer = self._cpu_twin(ModelVariant.TRANSFORMER)  # bf16

            to_dev = lambda x: x.to(dev)  # inputs already bf16 / int

            for i, t in enumerate(timesteps):
                logger.info("[STEP] transformer step {}/{}", i + 1, num_inference_steps)

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                timestep = t.expand(latent_model_input.shape[0])

                tt_inputs = [
                    to_dev(latent_model_input),
                    to_dev(timestep),
                    to_dev(prompt_embeds_t5),
                    to_dev(prompt_embeds_llama3),
                    to_dev(pooled_prompt_embeds),
                ]
                # Kept in bf16: the trajectory below must match the stock pipeline.
                noise_pred = tt_transformer(*tt_inputs).cpu()

                # CPU golden fed the same inputs the TT transformer saw.
                golden_noise = twin_transformer(
                    latent_model_input,
                    timestep,
                    prompt_embeds_t5,
                    prompt_embeds_llama3,
                    pooled_prompt_embeds,
                )
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

                # Advance the trajectory with the TT output (deployment behavior).
                # HiDream predicts the negated flow; the sign flip precedes guidance.
                noise_pred = -noise_pred
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

            transformer = transformer.to("cpu")
            del tt_transformer, transformer
            gc.collect()
            torch_xla.sync()
            logger.info("[STAGE] transformer: done")

            # VAE decode — CPU (bf16).
            logger.info("[STAGE] vae: CPU (not on TT)")
            vae = self._cpu_twin(ModelVariant.VAE)
            latents = (
                latents / vae.vae.config.scaling_factor
            ) + vae.vae.config.shift_factor
            image = vae(latents)
            logger.info("[STAGE] vae: done")
            return image


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.qb2_blackhole
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="HiDreamI1Full_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_hidream_e2e_pipeline():
    """HiDream-I1-Full pipeline — transformer on TT (bf16) with per-step PCC."""
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    config = HiDreamI1Config(device="cpu")
    pipeline = HiDreamI1Pipeline(config=config)
    pipeline.setup()
    pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )