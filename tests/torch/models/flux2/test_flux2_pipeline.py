# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — nightly PCC-gated text-to-image e2e test on Tenstorrent.

The standard diffusers ``Flux2Pipeline`` orchestrates the run (tokenizer +
scheduler stay on CPU), but every compute module runs on Tenstorrent, compiled
with ``torch.compile(backend="tt")`` and tensor-parallel sharded via the SPMD
shard specs from ``tt_forge_models``:

  - text encoder (Mistral3, ~24B)  → sharded
  - transformer  (Flux2, ~32B)     → sharded
  - VAE decoder  (~84M)            → replicated

Every stage is gated on PCC against a CPU twin fed the same inputs the device
saw: the prompt embeds once, the noise prediction on the first
``PCC_CHECK_STEPS`` denoise steps, and the decoded pixels once. The trajectory is
always advanced with the *device* output (deployment behavior), so a PCC drop
anywhere shows up as a test failure rather than a silently degraded image.

The transformer's CPU twin is a second ~32B module, so it is loaded lazily at the
first checked step and dropped once the checked steps are done — the remaining
steps then run device-only at the normal peak.

Memory strategy (peak ≈ max(component) rather than the sum):
  * Stage 1 runs the CPU golden on the encoder *before* placing it on device (no
    second copy), encodes the prompt on TT, then evicts the encoder.
  * Stage 2 routes the pipeline's transformer/VAE calls through compiled wrappers
    that move inputs to device and return CPU tensors each call, so the denoise
    loop keeps only one step's activations resident. The VAE is placed lazily at
    first decode (after the denoise loop) so it never inflates the denoise peak.
"""

import gc
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import Flux2Pipeline
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.flux2.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.flux2.pytorch.src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    PROMPT,
    REPO_ID,
    SEED,
    WIDTH,
    Mistral3TextEncoderWrapper,
    load_transformer,
    shard_text_encoder_specs,
    shard_transformer_specs,
    tokenize_prompt,
)

NUM_INFERENCE_STEPS = 50
# Denoise steps that get a CPU twin + PCC assert. The twin is another ~32B
# module and one CPU forward dominates the step time, so gate the leading steps
# (where a numerical break shows up first) instead of all 50.
PCC_CHECK_STEPS = 4

# Every stage clears the default, so no stage needs a relaxed gate.
PCC_THRESHOLD = 0.99

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _assert_pcc(stage: str, device_out, golden_out, threshold: float) -> float:
    pcc = _pcc(device_out, golden_out)
    logger.info("[PCC] {}: pcc={:.6f} (threshold {})", stage, pcc, threshold)
    assert pcc >= threshold, f"{stage} PCC {pcc:.6f} below threshold {threshold}"
    return pcc


class _PccDenoiser:
    """Routes Flux2Pipeline's transformer calls to the TP-sharded model on TT.

    The first ``PCC_CHECK_STEPS`` calls also run an fp-identical CPU twin on the
    same inputs and assert PCC on the noise prediction.
    """

    def __init__(self, transformer, mesh):
        self._dev = torch_xla.device()
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype
        self._step = 0
        self._twin = None
        self.pccs = []

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        specs = shard_transformer_specs(transformer)
        assert specs, "transformer shard spec is empty — would run replicated/OOM"
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(transformer, backend="tt")

    def _cpu_twin(self):
        # Loaded on first use so the second ~32B copy is only resident while the
        # checked steps run.
        if self._twin is None:
            logger.info("[load] CPU twin: transformer ({})", DTYPE)
            self._twin = load_transformer(DTYPE)
        return self._twin

    def _release_twin(self):
        if self._twin is not None:
            logger.info("[free] CPU twin: transformer (checked steps done)")
            self._twin = None
            gc.collect()

    def __call__(self, **kwargs):
        self._step += 1
        checked = self._step <= PCC_CHECK_STEPS

        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        out = self._compiled(**moved)
        # .cpu() is the sync point: it forces the pending graph to execute and
        # only returns once the result is on host, so no explicit sync is needed.
        if isinstance(out, (tuple, list)):
            result = type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        else:
            result = out.cpu()

        if checked:
            with torch.no_grad():
                golden = self._cpu_twin()(**kwargs)
            device_pred = result[0] if isinstance(result, (tuple, list)) else result
            golden_pred = golden[0] if isinstance(golden, (tuple, list)) else golden
            self.pccs.append(
                _assert_pcc(
                    f"transformer step {self._step}/{NUM_INFERENCE_STEPS}",
                    device_pred,
                    golden_pred,
                    PCC_THRESHOLD,
                )
            )
            if self._step == PCC_CHECK_STEPS:
                self._release_twin()

        return result


class _PccVAEDecoder:
    """Routes Flux2Pipeline's vae.decode() to TT (replicated), placed lazily.

    The pipeline reads ``vae.bn`` / ``vae.config`` / ``vae.dtype`` for the
    host-side batch-norm denorm, then calls
    ``vae.decode(latents, return_dict=False)[0]``. The CPU golden runs on the
    same module before it is placed on device, so no second copy is needed.
    """

    def __init__(self, vae, mesh):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.bn = vae.bn  # stays on CPU; pipeline reads it host-side for denorm
        self._vae = vae
        self._compiled = None
        self.pcc = None

    def decode(self, latents, return_dict=False):
        # CPU golden first — while the VAE is still on host, so the check costs
        # no extra copy.
        if self._compiled is None:
            with torch.no_grad():
                golden = self._vae.decode(latents, return_dict=False)[0]
            # Lazy device placement: keep the VAE off-device during the denoise
            # loop so it does not inflate the denoiser's peak DRAM.
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        else:
            golden = None

        # .cpu() forces the graph to execute and blocks until the result is on
        # host — the compiled lambda always returns a tensor, so no guard needed.
        image = self._compiled(latents.to(self._dev)).cpu()

        if golden is not None:
            self.pcc = _assert_pcc("vae decode", image, golden, PCC_THRESHOLD)

        return (image,)


class Flux2TTPipeline:
    """diffusers Flux2Pipeline with every module on TT, tensor-parallel sharded,
    each stage gated on PCC against a CPU twin."""

    def __init__(self, height: int = HEIGHT, width: int = WIDTH):
        self.height = height
        self.width = width

    def setup(self):
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        # Mesh from device count: the "model" axis carries the shard specs'
        # contraction-parallel degree, extra devices go to "batch".
        mesh_shape, mesh_names = ModelLoader(ModelVariant.TRANSFORMER).get_mesh_config(
            self.num_devices
        )
        self.mesh = get_mesh(mesh_shape, mesh_names)
        logger.info("[setup] mesh {} over {} device(s)", mesh_shape, self.num_devices)
        self.pipe = Flux2Pipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ):
        dev = torch_xla.device()

        # ── Stage 1: text encoder (sharded, compiled) → prompt embeds, evict ──
        logger.info("[STAGE] text_encoder (sharded): start")
        text_encoder = self.pipe.text_encoder
        encoder_wrapper = Mistral3TextEncoderWrapper(text_encoder).eval()
        input_ids, attention_mask = tokenize_prompt(prompt)

        # CPU golden on the host copy, before device placement — the twin costs
        # nothing extra because the module has not moved yet.
        with torch.no_grad():
            golden_embeds = encoder_wrapper(input_ids, attention_mask)

        text_encoder = text_encoder.to(dev)
        if hasattr(text_encoder, "tie_weights"):
            text_encoder.tie_weights()
        te_specs = shard_text_encoder_specs(text_encoder)
        assert te_specs, "text-encoder shard spec is empty — descent failed (would OOM)"
        for tensor, spec in te_specs.items():
            xs.mark_sharding(tensor, self.mesh, spec)
        te_compiled = torch.compile(encoder_wrapper, backend="tt")

        with torch.no_grad():
            prompt_embeds = te_compiled(input_ids.to(dev), attention_mask.to(dev))
        # .cpu() forces execution and blocks until the embeds are on host.
        prompt_embeds = prompt_embeds.cpu()
        _assert_pcc("text_encoder", prompt_embeds, golden_embeds, PCC_THRESHOLD)

        # Free the 24B encoder from device before placing the 32B denoiser.
        self.pipe.text_encoder = text_encoder.to("cpu")
        del te_compiled, encoder_wrapper, golden_embeds
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] text_encoder: done")

        # ── Stage 2: denoiser (sharded) + VAE (replicated, lazy) → image ─────
        logger.info(
            "[STAGE] transformer (sharded) + vae: start ({} steps, PCC on first {})",
            num_inference_steps,
            PCC_CHECK_STEPS,
        )
        self.pipe.transformer = _PccDenoiser(self.pipe.transformer, self.mesh)
        self.pipe.vae = _PccVAEDecoder(self.pipe.vae, self.mesh)

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        result = self.pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            height=self.height,
            width=self.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
        logger.info("[STAGE] transformer + vae: done")
        return result.images[0]


@pytest.mark.tensor_parallel
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.qb2_blackhole
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="Flux2_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_flux2_pipeline():
    """FLUX.2-dev pipeline — all modules on TT (sharded), every stage PCC-gated."""
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    output_path = "flux2_pipeline_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    pipeline = Flux2TTPipeline()
    pipeline.setup()
    image = pipeline.generate(
        PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED
    )
    image.save(output_path)

    assert output_file.exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
    logger.info(f"Output image saved to {output_path} ({width}x{height})")
