# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HiDream-I1-Full — nightly e2e pipeline test, every TT component PCC-gated
against a CPU twin in the same dtype. The CLIP-L and CLIP-G text encoders, the
Sparse-MoE MM-DiT transformer (tensor-parallel sharded) and the VAE decoder run
bf16 on TT; the T5 and Llama encoders and the scheduler stay on CPU. Each CLIP
encoder is checked once per prompt — twice under CFG, positive and negative — the
transformer once per denoising step, and the VAE once.

The pipeline itself lives in ``tt_forge_models`` and is shared with the runnable
demo and the benchmark; this test only wraps the TT components' forwards.
"""

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.hidream_i1.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.hidream_i1.pytorch.pipeline import (
    GUIDANCE_SCALE,
    NEGATIVE_PROMPT,
    PROMPT,
    SEED,
    TT_DTYPE,
    HiDreamI1Config,
    HiDreamI1Pipeline,
)

# 10 for now, will be bumped to 50 once the rest of the components are enabled on TT.
NUM_INFERENCE_STEPS = 10
PCC_THRESHOLD = 0.99

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _twin(variant: ModelVariant):
    """Load the CPU golden for a component, in the dtype it runs on TT."""
    return ModelLoader(variant).load_model(dtype_override=TT_DTYPE)


def _attach_pcc_checks(pipeline: HiDreamI1Pipeline) -> None:
    """Wrap every TT component's forward with a CPU-twin PCC check, asserted per
    forward. The pipeline keeps using the real TT output."""

    def attach(module, name, build_twin, pick=lambda out: out):
        orig_forward = module.forward
        twin = {"model": None}
        step = {"n": 0}

        def _cpu_twin():
            if twin["model"] is None:
                logger.info("[PCC] loading CPU twin: {}", name)
                twin["model"] = build_twin()
            return twin["model"]

        # The pipeline calls every component positionally, so the twin gets the
        # same args on CPU; a future kwargs call raises rather than mismatching.
        def wrapped_forward(*args):
            out = orig_forward(*args)
            device_sample = pick(out).to("cpu")
            golden = pick(_cpu_twin()(*[a.to("cpu") for a in args]))

            step["n"] += 1
            pcc = _pcc(device_sample, golden)
            logger.info("[PCC] {} forward {}: pcc={:.6f}", name, step["n"], pcc)
            assert (
                pcc >= PCC_THRESHOLD
            ), f"{name} forward {step['n']} PCC {pcc:.6f} below {PCC_THRESHOLD}"
            return out

        module.forward = wrapped_forward

    attach(
        pipeline.text_encoder,
        "text_encoder (CLIP-L)",
        lambda: _twin(ModelVariant.TEXT_ENCODER),
    )
    attach(
        pipeline.text_encoder_2,
        "text_encoder_2 (CLIP-G)",
        lambda: _twin(ModelVariant.TEXT_ENCODER_2),
    )
    # text_encoder_3 (T5) and text_encoder_4 (Llama) run on CPU, so there is
    # nothing to gate: https://github.com/tenstorrent/tt-xla/issues/6018 and
    # https://github.com/tenstorrent/tt-xla/issues/6019
    # The twin is the stock diffusers MoE — enable_sparse_mlp is applied to the
    # device copy only, so the golden exercises the unswapped expert path.
    attach(pipeline.transformer, "transformer", lambda: _twin(ModelVariant.TRANSFORMER))
    attach(pipeline.vae, "vae", lambda: _twin(ModelVariant.VAE))


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.qb2_blackhole
@pytest.mark.tensor_parallel
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="HiDreamI1Full_Pipeline",
    model_group=ModelGroup.RED,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_hidream_e2e_pipeline():
    """Run the HiDream-I1-Full pipeline with per-component PCC vs CPU twins."""
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    pipeline = HiDreamI1Pipeline(config=HiDreamI1Config())
    pipeline.setup()
    _attach_pcc_checks(pipeline)

    pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
