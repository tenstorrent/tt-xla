# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — nightly PCC-gated text-to-image e2e test on Tenstorrent.

The pipeline implementation is the shared one in ``tt_forge_models``, the same
code the demo (``examples/pytorch/flux2.py``) and the benchmark
(``tests/benchmark/test_imagegen.py::test_flux2``) run. This module only adds the
PCC gating, via the pipeline's substitution seams:

  - ``DENOISER_CLS`` / ``VAE_CLS`` swap in checking subclasses of the shared
    plain-callable wrappers,
  - ``_pre_place`` computes the text encoder's golden while the module is still
    on host, and ``_intercept`` compares the device result against it.

Nothing about staging, eviction or the compiled graphs is duplicated here, so
the test exercises the shipped pipeline rather than a copy that can drift from
it.

Every stage is gated on PCC against a CPU twin fed the same inputs the device
saw: the prompt embeds once, the noise prediction on the first
``PCC_CHECK_STEPS`` denoise steps, and the decoded pixels once. The trajectory is
always advanced with the *device* output (deployment behavior), so a PCC drop
anywhere shows up as a test failure rather than a silently degraded image.

Memory strategy (peak ≈ max(component) rather than the sum) is preserved: the
~24B encoder's golden runs on the host copy *before* placement so it costs no
second copy, the transformer's twin is loaded lazily at the first checked step
and dropped once the checked steps finish, and the VAE's golden runs before its
lazy placement.
"""

import gc
from pathlib import Path

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.flux2.pytorch.pipeline import (
    NUM_INFERENCE_STEPS,
    Flux2Config,
    Flux2TTPipeline,
    _DeviceDenoiser,
    _DeviceVAEDecoder,
    save_image,
)
from third_party.tt_forge_models.flux2.pytorch.src.model_utils import (
    DTYPE,
    HEIGHT,
    PROMPT,
    SEED,
    WIDTH,
    load_transformer,
)

# Denoise steps that get a CPU twin + PCC assert. The twin is a second copy of
# the ~32B denoiser and one CPU forward dominates the step time, so gate the
# leading steps (where a numerical break shows up first) instead of all 50.
PCC_CHECK_STEPS = 4

# Transformer steps and vae decode clear the default; the text encoder measures
# ~0.981 (TP Mistral3), so it gets its own gate.
PCC_THRESHOLD = 0.99
TEXT_ENCODER_PCC_THRESHOLD = 0.98

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _assert_pcc(stage: str, device_out, golden_out, threshold: float) -> float:
    pcc = float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))
    logger.info(f"[PCC] {stage}: pcc={pcc:.6f} (threshold {threshold})")
    assert pcc >= threshold, f"{stage} PCC {pcc:.6f} below threshold {threshold}"
    return pcc


class _PccEncoderCheck:
    """Wraps the text encoder's COMPILED callable and compares against a golden
    captured before placement (see ``_pre_place``)."""

    def __init__(self, compiled, pipeline):
        self._compiled = compiled
        self._pipeline = pipeline
        self._checked = False

    def __call__(self, *args, **kwargs):
        out = self._compiled(*args, **kwargs)
        golden = self._pipeline.goldens.get("text_encoder")
        if not self._checked and golden is not None:
            self._checked = True
            self._pipeline.pccs["text_encoder"] = _assert_pcc(
                "text_encoder", out.cpu().to(DTYPE), golden, TEXT_ENCODER_PCC_THRESHOLD
            )
        return out


class _PccDenoiser(_DeviceDenoiser):
    """Shared TP-sharded denoiser, PCC-checked on the first PCC_CHECK_STEPS calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._twin = None
        self.pccs = []

    def _cpu_twin(self):
        # Loaded on first use so the second copy is only resident while the
        # checked steps run.
        if self._twin is None:
            logger.info(f"[load] CPU twin: transformer ({DTYPE})")
            self._twin = load_transformer(DTYPE)
        return self._twin

    def __call__(self, **kwargs):
        self._step += 1
        result = super().__call__(**kwargs)
        if self._step <= PCC_CHECK_STEPS:
            with torch.no_grad():
                golden = self._cpu_twin()(**kwargs)
            dev_pred = result[0] if isinstance(result, (tuple, list)) else result
            gold_pred = golden[0] if isinstance(golden, (tuple, list)) else golden
            self.pccs.append(
                _assert_pcc(
                    f"transformer step {self._step}/{NUM_INFERENCE_STEPS}",
                    dev_pred,
                    gold_pred,
                    PCC_THRESHOLD,
                )
            )
            if self._step == PCC_CHECK_STEPS:
                logger.info("[free] CPU twin: transformer (checked steps done)")
                self._twin = None
                gc.collect()
        return result


class _PccVAEDecoder(_DeviceVAEDecoder):
    """Shared VAE decode, PCC-checked on its first decode.

    The golden runs on the host copy before the shared decode places it, so the
    check costs no second copy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pcc = None

    def decode(self, latents, return_dict=False):
        golden = None
        if self._compiled is None:  # first decode: VAE still on host
            with torch.no_grad():
                golden = self._vae.decode(latents, return_dict=False)[0]
        out = super().decode(latents, return_dict=return_dict)
        if golden is not None:
            self.pcc = _assert_pcc(
                "vae decode", self.last_pixels, golden, PCC_THRESHOLD
            )
        return out


class PccFlux2TTPipeline(Flux2TTPipeline):
    """The shipped pipeline with PCC checks on every stage.

    generate(), the staging and the warm machinery are all inherited.
    """

    DENOISER_CLS = _PccDenoiser
    VAE_CLS = _PccVAEDecoder

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goldens = {}
        self.pccs = {}

    def _pre_place(self, name, wrapper, *inputs):
        # Still on host: capture the golden here so the check needs no second copy
        # of the ~24B encoder.
        if name == "text_encoder":
            with torch.no_grad():
                self.goldens[name] = wrapper(*inputs).to(DTYPE)

    def _intercept(self, name, compiled):
        if name == "text_encoder":
            return _PccEncoderCheck(compiled, self)
        return compiled


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
    pcc=PCC_THRESHOLD,
)
def test_flux2_pipeline():
    """FLUX.2-dev pipeline — all modules on TT (sharded), every stage PCC-gated."""
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    output_path = "flux2_pipeline_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    # warm_iters defaults to 0: this test gates correctness, so it does not pay
    # for the in-residency repeats the benchmark uses.
    pipeline = PccFlux2TTPipeline(config=Flux2Config())
    pipeline.setup()
    pixels = pipeline.generate(
        PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED
    )
    # The shared pipeline returns raw pixels in [-1, 1]; save them as the demo does.
    save_image(pixels, output_path)

    assert output_file.exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
    logger.info(f"Output image saved to {output_path} ({width}x{height})")
