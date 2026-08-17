# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image — nightly e2e text-to-image pipeline with per-component PCC checks.

Every compute module runs on Tenstorrent, orchestrated by the diffusers
QwenImagePipeline: the Qwen2.5-VL text encoder and the QwenImage MMDiT
transformer are both tensor-parallel sharded on the mesh model axis, and the VAE
decoder is replicated. Each module is placed and evicted in turn so peak DRAM ≈
max(component). On the first TT forward of each component the same inputs are run
through a CPU twin and PCC is asserted against ``PCC_THRESHOLD``.
"""

import gc
from types import SimpleNamespace

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import QwenImagePipeline
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.qwen_image.pytorch.src.model_utils import (
    DTYPE,
    HEIGHT,
    MESH_NAMES,
    MESH_SHAPES,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    POSITIVE_MAGIC,
    PROMPT,
    REPO_ID,
    SEED,
    TOKENIZER_MAX_LENGTH,
    TRUE_CFG_SCALE,
    WIDTH,
    load_text_encoder,
    load_transformer,
    load_vae,
    shard_text_encoder_specs,
    shard_transformer_specs,
)

PCC_THRESHOLD = 0.99

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _assert_pcc(name: str, device_out, golden_out) -> None:
    pcc = float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))
    logger.info(f"[PCC] {name}: pcc={pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"{name} PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"


class _DeviceTextEncoder:
    """QwenImagePipeline text_encoder on TT (tensor-parallel sharded); one-shot PCC check."""

    def __init__(self, text_encoder, mesh):
        self._dev = torch_xla.device()
        self.dtype = next(text_encoder.parameters()).dtype
        self.config = text_encoder.config
        text_encoder = text_encoder.to(self._dev)
        if hasattr(text_encoder, "tie_weights"):
            text_encoder.tie_weights()
        # Tensor-parallel shard: replicated the encoder is ~16.6 GB/chip and
        # cannot coexist with the transformer (the stack does not free a compiled
        # module's device memory on eviction); sharding drops it to ~4 GB/chip.
        for tensor, spec in shard_text_encoder_specs(text_encoder).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(text_encoder, backend="tt")
        self._checked = False

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True):
        out = self._compiled(
            input_ids=input_ids.to(self._dev),
            attention_mask=(
                attention_mask.to(self._dev) if attention_mask is not None else None
            ),
            output_hidden_states=True,
        )
        last_hidden = out.hidden_states[-1].cpu()
        if not self._checked:
            self._checked = True
            twin = load_text_encoder(DTYPE)
            with torch.no_grad():
                golden = twin(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                ).hidden_states[-1]
            _assert_pcc("text_encoder", last_hidden, golden)
            del twin
            gc.collect()
        # Only hidden_states[-1] is consumed downstream.
        return SimpleNamespace(hidden_states=(last_hidden,))


class _DeviceDenoiser:
    """QwenImagePipeline transformer on TT (TP-sharded); one-shot PCC check."""

    def __init__(self, transformer, mesh):
        self._dev = torch_xla.device()
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype
        self.cache_context = transformer.cache_context

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        for tensor, spec in shard_transformer_specs(transformer).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(transformer, backend="tt")
        self._checked = False

    def __call__(self, **kwargs):
        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        # return_dict=False -> transformer returns a 1-tuple (sample,); .cpu()
        # forces graph execution and blocks until the result is on host.
        (sample,) = self._compiled(**moved)
        sample = sample.cpu()
        if not self._checked:
            self._checked = True
            twin = load_transformer(DTYPE)
            with torch.no_grad():
                (golden,) = twin(**kwargs)
            _assert_pcc("transformer", sample, golden)
            del twin
            gc.collect()
        return (sample,)


class _DeviceVAEDecoder:
    """QwenImagePipeline vae.decode() on TT (replicated); one-shot PCC check."""

    def __init__(self, vae):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.temperal_downsample = vae.temperal_downsample
        self._vae = vae
        self._compiled = None
        self._checked = False

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep the VAE off-device during the denoise loop
        # so it never coexists with the transformer's peak.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        image = self._compiled(latents.to(self._dev)).cpu()
        if not self._checked:
            self._checked = True
            twin = load_vae(DTYPE)
            with torch.no_grad():
                golden = twin.decode(latents, return_dict=False)[0]
            _assert_pcc("vae", image, golden)
            del twin
            gc.collect()
        return (image,)


class QwenImageTTPipeline:
    """diffusers QwenImagePipeline with every compute module on TT."""

    def setup(self):
        enable_spmd()
        self.mesh = get_mesh(MESH_SHAPES[xr.global_runtime_device_count()], MESH_NAMES)
        self.pipe = QwenImagePipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    def generate(self, prompt, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED):
        # Stage 1: text encoder → prompt embeds (host-side masked extraction runs
        # on CPU, so encode on CPU tensors), then evict before the transformer.
        logger.info("[STAGE] Text encoder: start")
        text_encoder = self.pipe.text_encoder
        te_wrapper = _DeviceTextEncoder(text_encoder, self.mesh)
        self.pipe.text_encoder = te_wrapper
        cpu = torch.device("cpu")
        prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
            prompt=prompt + POSITIVE_MAGIC,
            device=cpu,
            num_images_per_prompt=1,
            max_sequence_length=TOKENIZER_MAX_LENGTH,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = self.pipe.encode_prompt(
            prompt=NEGATIVE_PROMPT,
            device=cpu,
            num_images_per_prompt=1,
            max_sequence_length=TOKENIZER_MAX_LENGTH,
        )
        # Evict the text encoder before placing the transformer (flux2 pattern).
        self.pipe.text_encoder = text_encoder.to("cpu")
        del te_wrapper
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] Text encoder: done")

        # Stage 2: transformer (sharded) + VAE (replicated) → image.
        logger.info("[STAGE] Transformer + VAE: start")
        self.pipe.transformer = _DeviceDenoiser(self.pipe.transformer, self.mesh)
        self.pipe.vae = _DeviceVAEDecoder(self.pipe.vae)

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        result = self.pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
        )
        logger.info("[STAGE] Transformer + VAE: done")
        return result.images[0]


@pytest.mark.tensor_parallel
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.qb2_blackhole
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="QwenImage_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
    pcc=PCC_THRESHOLD,
)
def test_qwen_image_pipeline():
    """Full Qwen-Image pipeline on TT with per-component PCC gating."""
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    pipeline = QwenImageTTPipeline()
    pipeline.setup()
    pipeline.generate(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED)
