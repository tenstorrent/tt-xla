# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image -- nightly PCC-gated text-to-image e2e test on Tenstorrent.

Runs the shared ``tt_forge_models`` pipeline, the same code the demo and the
benchmark use. PCC gating is added by subclassing each device wrapper to check
its first TT forward against a CPU twin, and the pipeline to swap those in.
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


_CHECKED = []


def _assert_pcc(name: str, device_out, golden_out) -> None:
    pcc = float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))
    logger.info(f"[PCC] {name}: pcc={pcc:.6f}")
    _CHECKED.append(name)
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
    """Shared VAE decode, PCC-checked on its first decode. The shipped decode slices
    the singleton temporal dim in-graph, so the twin's 5D output is sliced to match."""

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

    # warm_iters defaults to 0: this test gates correctness, so it does not pay
    # for the in-residency repeats the benchmark uses.
    pipeline = PccQwenImagePipeline(config=QwenImageConfig())
    pipeline.setup()
    pipeline.generate(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED)

    # Without this the test would pass having verified nothing if the wrappers
    # ever stopped being swapped in.
    assert _CHECKED, "no PCC checks ran: the checking wrappers never fired"
