# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mochi-1 preview — nightly e2e pipeline test, every TT component PCC-gated
against a CPU twin in the same dtype. The T5-XXL text encoder (fp32) and the DiT
(bf16) both run tensor-parallel on TT; scheduler and VAE stay on CPU. The encoder
is checked on each of its two forwards (cond + uncond), the DiT once per
denoising step.

CFG is on (guidance_scale=4.5), so every DiT forward — and every twin forward —
sees a batch-2 cat([uncond, cond]) input.
"""

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.mochi.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.mochi.pytorch.src.pipeline import (
    TEXT_ENCODER_DTYPE,
    Mochi1Config,
    Mochi1Pipeline,
)

PROMPT = (
    "Close-up of a chameleon's eye, with its scaly skin changing color. "
    "Ultra high resolution 4k."
)
SEED = 0
NUM_INFERENCE_STEPS = 10
NUM_FRAMES = 24
# Set by the text encoder, which tops out ~0.95 even in fp32 (0.9494 cond /
# 0.9518 uncond); the DiT runs ~1.0. See
# https://github.com/tenstorrent/tt-xla/issues/5995
PCC_THRESHOLD = 0.94

VARIANT_NAME = ModelVariant.MOCHI
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _cpu(x):
    return x.to("cpu") if isinstance(x, torch.Tensor) else x


def _attach_pcc_checks(pipeline: Mochi1Pipeline) -> None:
    """Wrap every TT component's forward with an inline CPU-twin PCC check,
    asserted per forward so a diverging step fails fast; the pipeline continues
    using the real TT output regardless."""

    def attach(module, name, subfolder, dtype, pick=lambda out: out):
        orig_forward = module.forward
        twin = {"model": None}
        step = {"n": 0}

        def _cpu_twin():
            if twin["model"] is None:
                logger.info("[PCC] loading {} CPU twin: {}", dtype, name)
                twin["model"] = ModelLoader(
                    VARIANT_NAME, subfolder=subfolder
                ).load_model(dtype_override=dtype)
            return twin["model"]

        # The twin sees the same args, positional or keyword, moved to host.
        def wrapped_forward(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            device_sample = pick(out).to("cpu")
            golden = pick(
                _cpu_twin()(
                    *[_cpu(a) for a in args],
                    **{k: _cpu(v) for k, v in kwargs.items()},
                )
            )

            step["n"] += 1
            pcc = _pcc(device_sample, golden)
            logger.info("[PCC] {} forward {}: pcc={:.6f}", name, step["n"], pcc)
            assert (
                pcc >= PCC_THRESHOLD
            ), f"{name} forward {step['n']} PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"
            return out

        module.forward = wrapped_forward

    # Each twin runs in the dtype its TT counterpart runs in.
    attach(
        pipeline.text_encoder,
        "text_encoder",
        "text_encoder",
        TEXT_ENCODER_DTYPE,
        pick=lambda out: out[0],
    )
    attach(
        pipeline.transformer,
        "dit",
        "transformer",
        torch.bfloat16,
        pick=lambda out: out[0] if isinstance(out, (tuple, list)) else out,
    )


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.tensor_parallel
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_pipeline():
    """Run the Mochi-1 pipeline with per-component PCC vs same-dtype CPU twins."""
    xr.set_device_type("TT")

    pipeline = Mochi1Pipeline(
        config=Mochi1Config(
            num_inference_steps=NUM_INFERENCE_STEPS, num_frames=NUM_FRAMES
        )
    )
    pipeline.setup()
    _attach_pcc_checks(pipeline)

    pipeline.generate(prompt=PROMPT, seed=SEED)
