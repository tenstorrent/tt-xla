# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — PCC-gated nightly e2e text-to-image pipeline test (multichip).

The diffusers ``FluxPipeline`` orchestrates the run (tokenizers, scheduler and
latent bookkeeping stay on CPU) at the source geometry/sampling params
(1024x1024, 50 steps, guidance 3.5, seq-512). Every compute module runs on
Tenstorrent via ``torch.compile(backend="tt")``:

  - CLIP text encoder (CLIPTextModel)      → replicated
  - T5   text encoder (T5EncoderModel)     → replicated
  - transformer (FluxTransformer2DModel)   → tensor-parallel sharded (model axis)
  - VAE decoder (AutoencoderKL)            → replicated

This single test replaces the four standalone component tests (CLIP / T5 /
transformer / VAE). It follows the SDXL-Lightning / Playground-v2.5 pattern: as
each component runs on device, its output is compared (PCC) against a CPU
"golden" fed the same input, and the test fails fast the moment any component
drops below its threshold. The pipeline itself keeps running with the TT outputs
(real deployment behaviour); the PCC check is a side-channel assertion.

Memory strategy — peak DRAM ≈ max(component), i.e. the same as any one of the
old standalone component tests:
  * Each component's CPU golden is computed while it is still on host, *before*
    it is placed on device, so the device never holds a component twice.
  * Stage 1 places CLIP → uses → evicts, then T5 → uses → evicts.
  * Stage 2 places the transformer lazily (first denoise step), evicts it before
    the VAE decode, and places the VAE lazily. Only one big module is resident on
    TT DRAM at a time.

Device count: forced to 4 chips (mesh ``(1, 4)`` — degree-4 tensor parallel, no
wasted batch axis) via ``TT_VISIBLE_DEVICES`` even on an 8-chip lb-blackhole,
matching the FLUX.1 component tests. Must run in an isolated process (``--forked``)
so the env var is read at first device init.
"""

import gc
import os
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

from third_party.tt_forge_models.flux.pytorch.src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    MAX_SEQUENCE_LENGTH,
    MESH_NAMES,
    MESH_SHAPES,
    PROMPT,
    REPO_ID,
    SEED,
    WIDTH,
    ClipTextEncoderWrapper,
    T5TextEncoderWrapper,
    shard_transformer_specs,
    tokenize_clip,
    tokenize_t5,
)

NUM_INFERENCE_STEPS = 50

# Per-component PCC thresholds.
#   * CLIP / transformer / VAE hold the 0.99 PccConfig default.
#   * T5 is relaxed to 0.95: accumulative bf16 device-matmul drift across the 24
#     encoder blocks lands it just under 0.99 (same relaxation the standalone T5
#     component test used); it does not affect the real-input e2e image.
PCC_CLIP = 0.99
PCC_T5 = 0.95
PCC_TRANSFORMER = 0.99
PCC_VAE = 0.99


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _first_tensor(out):
    """Flux modules called with return_dict=False return a 1-tuple; unwrap it."""
    return out[0] if isinstance(out, (tuple, list)) else out


class _DeviceDenoiser:
    """Routes FluxPipeline's transformer calls to the TP-sharded model on TT.

    The transformer stays on host until the first denoise step, so its CPU golden
    can be computed before it ever lands on device (keeping peak DRAM ≈ one
    transformer). PCC is gated once, on the first step — the old standalone
    transformer component test also validated a single forward.
    """

    def __init__(self, transformer, mesh):
        self._dev = torch_xla.device()
        self._mesh = mesh
        self._transformer = transformer  # on CPU until first call
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype
        self._compiled = None
        self.pcc = None

    @contextmanager
    def cache_context(self, *args, **kwargs):
        # FluxPipeline wraps the forward in `with transformer.cache_context(...)`
        # (diffusers CacheMixin); we don't cache, so this is a no-op.
        yield

    def _place(self):
        transformer = self._transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        for tensor, spec in shard_transformer_specs(transformer).items():
            xs.mark_sharding(tensor, self._mesh, spec)
        self._transformer = transformer
        self._compiled = torch.compile(transformer, backend="tt")

    def evict(self):
        """Move the transformer off device so the VAE decode peaks at ~= VAE."""
        if self._compiled is not None:
            self._transformer = self._transformer.to("cpu")
            self._compiled = None
            gc.collect()
            torch_xla.sync()

    def __call__(self, **kwargs):
        if self._compiled is None:
            # First step: CPU golden BEFORE device placement.
            cpu_kwargs = {
                k: (v.cpu() if torch.is_tensor(v) else v) for k, v in kwargs.items()
            }
            with torch.no_grad():
                golden = _first_tensor(self._transformer(**cpu_kwargs))

            self._place()
            moved = {
                k: (v.to(self._dev) if torch.is_tensor(v) else v)
                for k, v in kwargs.items()
            }
            out = self._compiled(**moved)
            # .cpu() forces the pending graph to execute and blocks until the
            # result is on host, so no explicit sync is needed.
            if isinstance(out, (tuple, list)):
                out = type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
            else:
                out = out.cpu()

            self.pcc = _pcc(_first_tensor(out), golden)
            logger.info(f"[PCC] transformer (step 1): pcc={self.pcc:.6f}")
            assert (
                self.pcc >= PCC_TRANSFORMER
            ), f"transformer PCC {self.pcc:.6f} below threshold {PCC_TRANSFORMER}"
            del golden
            gc.collect()
            return out

        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        out = self._compiled(**moved)
        # .cpu() is the sync point: it forces the pending graph to execute and
        # only returns once the result lands on host.
        if isinstance(out, (tuple, list)):
            return type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        return out.cpu()


class _DeviceVAEDecoder:
    """Routes FluxPipeline's vae.decode() to TT (replicated), placed lazily.

    Evicts the transformer (via ``denoiser``) before placing the VAE.
    """

    def __init__(self, vae, mesh, denoiser=None):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self._vae = vae
        self._denoiser = denoiser
        self._compiled = None
        self.pcc = None

    def decode(self, latents, return_dict=False):
        # CPU golden first (VAE is small), while still on host.
        with torch.no_grad():
            golden = self._vae.decode(latents.cpu(), return_dict=False)[0]

        # Free the transformer before the VAE lands on device.
        if self._denoiser is not None:
            self._denoiser.evict()

        # Lazy device placement: keep the VAE off-device during the denoise loop
        # so it does not inflate the denoiser's peak DRAM; place it only now.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        # .cpu() forces the graph to execute and blocks until the result is on host.
        out = self._compiled(latents.to(self._dev))
        image = out.cpu()

        self.pcc = _pcc(image, golden)
        logger.info(f"[PCC] vae: pcc={self.pcc:.6f}")
        assert self.pcc >= PCC_VAE, f"vae PCC {self.pcc:.6f} below threshold {PCC_VAE}"
        return (image,)


class FluxTTPipeline:
    """diffusers FluxPipeline with every module on TT, transformer TP-sharded.

    Each component's TT output is PCC-gated against a CPU golden fed the same
    input, so this one e2e run also covers what the standalone component tests did.
    """

    def __init__(self, height: int = HEIGHT, width: int = WIDTH):
        self.height = height
        self.width = width

    def setup(self):
        # Enables SPMD + shardy annotations; required so the StableHLO handed to
        # tt-mlir carries the @Sharding custom calls the presharded args need.
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        # "model" axis is degree 4 (the shard specs' TP degree); extra devices
        # go to the replicated "batch" axis.
        self.mesh = get_mesh(MESH_SHAPES[self.num_devices], MESH_NAMES)
        logger.info(
            f"FLUX.1 mesh {MESH_SHAPES[self.num_devices]} (names={MESH_NAMES}) "
            f"over {self.num_devices} devices"
        )
        self.pipe = FluxPipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    def _encode(self, wrapper_cls, module, input_ids, dev, pcc_threshold, name):
        """Place a replicated text encoder on device, encode, evict; PCC-gate it.

        The CPU golden is computed while the module is still on host, before it is
        moved (not copied) to device, so peak DRAM stays ≈ one encoder.
        """
        wrapper = wrapper_cls(module).eval()
        with torch.no_grad():
            golden = wrapper(input_ids)

        module = module.to(dev)
        compiled = torch.compile(wrapper, backend="tt")
        with torch.no_grad():
            out = compiled(input_ids.to(dev))
        torch_xla.sync()
        out = out.cpu().to(DTYPE)
        module = module.to("cpu")

        pcc = _pcc(out, golden)
        logger.info(f"[PCC] {name}: pcc={pcc:.6f}")
        assert pcc >= pcc_threshold, (
            f"{name} PCC {pcc:.6f} below threshold {pcc_threshold}"
        )
        del compiled, wrapper, golden
        return module, out

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ):
        dev = torch_xla.device()

        # ── Stage 1: text encoders (CLIP + T5, replicated) → embeds, then evict ──
        logger.info("[STAGE] CLIP text encoder: start")
        self.pipe.text_encoder, pooled_prompt_embeds = self._encode(
            ClipTextEncoderWrapper,
            self.pipe.text_encoder,
            tokenize_clip(prompt),
            dev,
            PCC_CLIP,
            "clip_text_encoder",
        )
        logger.info("[STAGE] CLIP text encoder: done")

        logger.info("[STAGE] T5 text encoder: start")
        self.pipe.text_encoder_2, prompt_embeds = self._encode(
            T5TextEncoderWrapper,
            self.pipe.text_encoder_2,
            tokenize_t5(prompt, max_sequence_length=MAX_SEQUENCE_LENGTH),
            dev,
            PCC_T5,
            "t5_text_encoder",
        )
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] T5 text encoder: done")

        # ── Stage 2: transformer (sharded) + VAE (replicated, lazy) → image ─────
        logger.info("[STAGE] Transformer + VAE: start")
        denoiser = _DeviceDenoiser(self.pipe.transformer, self.mesh)
        self.pipe.transformer = denoiser
        self.pipe.vae = _DeviceVAEDecoder(self.pipe.vae, self.mesh, denoiser=denoiser)

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
        logger.info("[STAGE] Transformer + VAE: done")
        return result.images[0]


@pytest.mark.tensor_parallel
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.lb_blackhole
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="Flux1_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_flux_pipeline():
    """Full FLUX.1-dev pipeline on TT (4-chip sharded) with per-component PCC gates."""
    os.environ.setdefault("TT_VISIBLE_DEVICES", "0,1,2,3")
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
