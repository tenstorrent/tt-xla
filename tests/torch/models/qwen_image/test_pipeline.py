# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image — nightly e2e text-to-image pipeline with per-component PCC checks.

The pipeline implementation is the shared one in ``tt_forge_models``, the same
code the demo (``examples/pytorch/qwen_image.py``) and the benchmark
(``tests/benchmark/test_imagegen.py``) run. This module only adds the PCC
gating: each device wrapper is subclassed to run the component's first TT
forward through a CPU twin and assert PCC against ``PCC_THRESHOLD``, and the
pipeline is subclassed to swap those wrappers in. Nothing about staging,
eviction or the compiled graphs is duplicated here, so the test exercises the
shipped pipeline rather than a copy that can drift from it.
"""

import gc

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.qwen_image.pytorch.pipeline import (
    QwenImageConfig,
    QwenImagePipeline,
    _DeviceDenoiser,
    _DeviceTextEncoder,
    _DeviceVAEDecoder,
)
from third_party.tt_forge_models.qwen_image.pytorch.src.model_utils import (
    DTYPE,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    load_text_encoder,
    load_transformer,
    load_vae,
)

PCC_THRESHOLD = 0.99

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _assert_pcc(name: str, device_out, golden_out) -> None:
    pcc = float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))
    logger.info(f"[PCC] {name}: pcc={pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"{name} PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"


class _PccTextEncoder(_DeviceTextEncoder):
    """Shared sharded text encoder, PCC-checked on its first forward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checked = False

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True):
        out = super().__call__(input_ids, attention_mask, output_hidden_states)
        if not self._checked:
            self._checked = True
            # The twin is a fresh CPU copy: same inputs, no `tt` backend attached.
            twin = load_text_encoder(DTYPE)
            with torch.no_grad():
                golden = twin(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                ).hidden_states[-1]
            _assert_pcc("text_encoder", out.hidden_states[-1], golden)
            del twin
            gc.collect()
        return out


class _PccDenoiser(_DeviceDenoiser):
    """Shared sharded transformer, PCC-checked on its first forward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checked = False

    def __call__(self, **kwargs):
        (sample,) = super().__call__(**kwargs)
        if not self._checked:
            self._checked = True
            # kwargs are still the host tensors; the parent moved its own copies.
            twin = load_transformer(DTYPE)
            with torch.no_grad():
                (golden,) = twin(**kwargs)
            _assert_pcc("transformer", sample, golden)
            del twin
            gc.collect()
        return (sample,)


class _PccVAEDecoder(_DeviceVAEDecoder):
    """Shared VAE decode, PCC-checked on its first decode.

    The shipped decode slices the singleton temporal dim in-graph, so the device
    result is 4D ``(B, 3, H, W)``; the CPU twin's 5D output is sliced the same way
    before comparing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checked = False

    def decode(self, latents, return_dict=False):
        out = super().decode(latents, return_dict=return_dict)
        if not self._checked:
            self._checked = True
            twin = load_vae(DTYPE)
            with torch.no_grad():
                golden = twin.decode(latents, return_dict=False)[0][:, :, 0]
            _assert_pcc("vae", self.last_pixels, golden)
            del twin
            gc.collect()
        return out


class PccQwenImagePipeline(QwenImagePipeline):
    """The shipped pipeline with PCC-checking wrappers swapped in."""

    TEXT_ENCODER_CLS = _PccTextEncoder
    DENOISER_CLS = _PccDenoiser
    VAE_CLS = _PccVAEDecoder


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

    # warm_iters=1: this test gates correctness, so there is no reason to pay for
    # the extra warm repeats the benchmark uses to measure steady-state cost.
    pipeline = PccQwenImagePipeline(config=QwenImageConfig(warm_iters=1))
    pipeline.setup()
    pipeline.generate(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED)
