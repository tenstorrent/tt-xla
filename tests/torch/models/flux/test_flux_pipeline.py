# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — nightly PCC-gated text-to-image e2e test on Tenstorrent.

The diffusers ``FluxPipeline`` orchestrates the run (tokenizers, scheduler and
latent bookkeeping stay on CPU) at the source geometry / sampling params
(1024x1024, 50 steps, guidance 3.5, seq-512). Every compute module runs on
Tenstorrent via ``torch.compile(backend="tt")``:

  - CLIP text encoder (CLIPTextModel)      → replicated
  - T5   text encoder (T5EncoderModel)     → replicated
  - transformer (FluxTransformer2DModel)   → tensor-parallel sharded (model axis)
  - VAE decoder (AutoencoderKL)            → replicated

Every stage is gated on PCC against a CPU twin fed the same inputs the device
saw: both prompt-embed streams once, the noise prediction on the first
``PCC_CHECK_STEPS`` denoise steps, and the decoded pixels once. The trajectory is
always advanced with the *device* output (deployment behavior), so a PCC drop
anywhere shows up as a test failure rather than a silently degraded image.

The transformer's CPU twin is a second full copy of the denoiser, so it is loaded
lazily at the first checked step and dropped once the checked steps are done —
the remaining steps then run device-only at the normal peak.

Memory strategy (peak ≈ max(component) rather than the sum): each encoder's CPU
golden runs on the host copy *before* that encoder is placed on device (so it
costs no second copy), the encoders are placed → used → evicted before the
transformer is placed, and the VAE is placed lazily at first decode.
"""

import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import FluxPipeline
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.flux.pytorch.src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    MAX_SEQUENCE_LENGTH,
    PROMPT,
    REPO_ID,
    SEED,
    WIDTH,
    ClipTextEncoderWrapper,
    T5TextEncoderWrapper,
    load_transformer,
    shard_transformer_specs,
    tokenize_clip,
    tokenize_t5,
)

NUM_INFERENCE_STEPS = 50
# Denoise steps that get a CPU twin + PCC assert. The twin is a second copy of
# the denoiser and one CPU forward dominates the step time, so gate the leading
# steps (where a numerical break shows up first) instead of all 50.
PCC_CHECK_STEPS = 4

# Only the T5 stream needs a relaxed gate; every other stage clears the default.
PCC_THRESHOLD_CLIP = 0.99
PCC_THRESHOLD_T5 = 0.95
PCC_THRESHOLD_TRANSFORMER = 0.99
PCC_THRESHOLD_VAE = 0.99

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
    """Routes FluxPipeline's transformer calls to the TP-sharded model on TT.

    The first ``PCC_CHECK_STEPS`` calls also run a CPU twin on the same inputs and
    assert PCC on the noise prediction.
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

    @contextmanager
    def cache_context(self, *args, **kwargs):
        # FluxPipeline wraps the forward in `with transformer.cache_context(...)`
        # (diffusers CacheMixin); we don't cache, so this is a no-op.
        yield

    def _cpu_twin(self):
        # Loaded on first use so the second copy is only resident while the
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
        # only returns once the result lands on host.
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
                    PCC_THRESHOLD_TRANSFORMER,
                )
            )
            if self._step == PCC_CHECK_STEPS:
                self._release_twin()

        return result


class _PccVAEDecoder:
    """Routes FluxPipeline's vae.decode() to TT (replicated), placed lazily.

    The CPU golden runs on the same module before it is placed on device, so no
    second copy is needed.
    """

    def __init__(self, vae, mesh):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
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
            self.pcc = _assert_pcc("vae decode", image, golden, PCC_THRESHOLD_VAE)

        return (image,)


class FluxTTPipeline:
    """diffusers FluxPipeline with every module on TT, transformer TP-sharded,
    each stage gated on PCC against a CPU twin."""

    def __init__(self, height: int = HEIGHT, width: int = WIDTH):
        self.height = height
        self.width = width

    def setup(self):
        # Enables SPMD + shardy annotations; required so the StableHLO handed to
        # tt-mlir carries the @Sharding custom calls the presharded args need.
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        # "model" axis carries the shard specs' TP degree; extra devices go to
        # the replicated "batch" axis.
        mesh_shape, mesh_names = ModelLoader(ModelVariant.TRANSFORMER).get_mesh_config(
            self.num_devices
        )
        self.mesh = get_mesh(mesh_shape, mesh_names)
        logger.info("[setup] mesh {} over {} device(s)", mesh_shape, self.num_devices)
        self.pipe = FluxPipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    def _encode_checked(self, stage, wrapper_cls, module, input_ids, dev, threshold):
        """CPU golden on the host copy, then place → encode → evict on device.

        Running the golden before placement means the check needs no second copy
        of the encoder. Returns ``(cpu_module, embeds)``.
        """
        logger.info("[STAGE] {}: start", stage)
        wrapper = wrapper_cls(module).eval()
        with torch.no_grad():
            golden = wrapper(input_ids)

        module = module.to(dev)
        compiled = torch.compile(wrapper, backend="tt")
        with torch.no_grad():
            out = compiled(input_ids.to(dev))
        torch_xla.sync()
        out = out.cpu().to(DTYPE)
        _assert_pcc(stage, out, golden, threshold)

        module = module.to("cpu")
        del compiled, wrapper, golden
        logger.info("[STAGE] {}: done", stage)
        return module, out

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ):
        dev = torch_xla.device()

        # ── Stage 1: text encoders (CLIP + T5, replicated) → embeds, evict ────
        self.pipe.text_encoder, pooled_prompt_embeds = self._encode_checked(
            "clip_text_encoder",
            ClipTextEncoderWrapper,
            self.pipe.text_encoder,
            tokenize_clip(prompt),
            dev,
            PCC_THRESHOLD_CLIP,
        )
        self.pipe.text_encoder_2, prompt_embeds = self._encode_checked(
            "t5_text_encoder",
            T5TextEncoderWrapper,
            self.pipe.text_encoder_2,
            tokenize_t5(prompt, max_sequence_length=MAX_SEQUENCE_LENGTH),
            dev,
            PCC_THRESHOLD_T5,
        )
        gc.collect()
        torch_xla.sync()

        # ── Stage 2: transformer (sharded) + VAE (replicated, lazy) → image ───
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
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=self.height,
            width=self.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
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
    model_name="Flux1_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_flux_pipeline():
    """FLUX.1-dev pipeline — all modules on TT (sharded), every stage PCC-gated."""
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    output_path = "flux1_pipeline_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    pipeline = FluxTTPipeline()
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
