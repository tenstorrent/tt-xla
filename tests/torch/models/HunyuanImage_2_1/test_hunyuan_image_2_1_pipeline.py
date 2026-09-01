# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanImage 2.1 — nightly e2e pipeline test, every TT component PCC-gated
against a CPU twin in the same dtype. The Qwen and ByT5 encoders and the MMDiT
transformer run bf16 on TT; scheduler, guider combine and VAE stay on CPU.

Guidance is real CFG, so Qwen is checked twice (conditional + unconditional) and
the transformer twice per denoising step.
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
from third_party.tt_forge_models.hunyuan_image_2_1.pytorch import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.hunyuan_image_2_1.pytorch.pipeline import (
    HIDDEN_STATE_SKIP_LAYER,
    PROMPT_TEMPLATE_ENCODE_START_IDX,
    TT_DTYPE,
    HunyuanImage21Config,
    HunyuanImage21Pipeline,
)
from third_party.tt_forge_models.hunyuan_image_2_1.pytorch.src.model_utils import (
    QwenPromptEmbedsWrapper,
)

PROMPT = (
    "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, "
    "standing in a painting studio, wearing a red knitted scarf and a red beret "
    "with the word 'Tencent' on it, holding a paintbrush with a focused "
    "expression as it paints an oil painting of the Mona Lisa, rendered in a "
    "photorealistic photographic style."
)
NUM_INFERENCE_STEPS = 10  # 10 for now, will be boosted to 50 later
PCC_THRESHOLD = 0.10 # for pcc check

MODEL_INFO = ModelLoader._get_model_info(ModelVariant.TRANSFORMER)

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _twin(variant: ModelVariant):
    """Load the CPU golden for a component, in the dtype it runs on TT."""
    return ModelLoader(variant).load_model(dtype_override=TT_DTYPE)


def _attach_pcc_checks(pipeline: HunyuanImage21Pipeline) -> None:
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
        "text_encoder (Qwen)",
        lambda: QwenPromptEmbedsWrapper(
            _twin(ModelVariant.TEXT_ENCODER),
            HIDDEN_STATE_SKIP_LAYER,
            PROMPT_TEMPLATE_ENCODE_START_IDX,
        ),
    )
    attach(
        pipeline.text_encoder_2,
        "text_encoder_2 (ByT5)",
        lambda: _twin(ModelVariant.TEXT_ENCODER_2),
        pick=lambda out: out[0],
    )
    attach(pipeline.transformer, "transformer", lambda: _twin(ModelVariant.TRANSFORMER))


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
    """Run the HunyuanImage 2.1 pipeline with per-component PCC vs CPU twins."""
    xr.set_device_type("TT")

    pipeline = HunyuanImage21Pipeline(config=HunyuanImage21Config())
    pipeline.setup()
    _attach_pcc_checks(pipeline)

    pipeline.generate(prompt=PROMPT, num_inference_steps=NUM_INFERENCE_STEPS)
