# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo 1.5 (480p t2v base) — nightly e2e pipeline test, every TT
component PCC-gated against a CPU twin in the same dtype. The Qwen2.5-VL and
ByT5 encoders and the DiT run bf16 on TT; scheduler and VAE stay on CPU.

Guidance is real CFG, so Qwen is checked twice (cond + uncond) and the DiT twice
per denoising step; ByT5 once, since the negative prompt's glyph stream is zeros.
"""

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from utils import BringupStatus, Category, incorrect_result

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.hunyuan_1_5.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.hunyuan_1_5.pytorch.src.model_utils import (
    QwenPromptEmbedsWrapper,
)
from third_party.tt_forge_models.hunyuan_1_5.pytorch.src.pipeline import (
    HIDDEN_STATE_SKIP_LAYER,
    PROMPT_TEMPLATE_ENCODE_START_IDX,
    HunyuanVideo15Config,
    HunyuanVideo15Pipeline,
)

# The double-quoted span is what routes text through text_encoder_2 (the ByT5
# glyph encoder); without it the pipeline feeds the DiT zero glyph embeds.
PROMPT = 'A girl holding a paper with words "Hello, world!"'
SEED = 42
NUM_INFERENCE_STEPS = 10
NUM_FRAMES = 25
PCC_THRESHOLD = 0.90

VARIANT_NAME = ModelVariant.TRANSFORMER
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _twin(variant: ModelVariant):
    """Load the CPU golden for a component, in the dtype it runs on TT."""
    return ModelLoader(variant).load_model(dtype_override=torch.bfloat16)


def _attach_pcc_checks(pipeline: HunyuanVideo15Pipeline) -> None:
    """Wrap every TT component's forward with a CPU-twin PCC check, asserted per
    forward so a diverging step fails fast. The pipeline keeps using the real TT
    output."""

    def attach(module, name, build_twin, pick=lambda out: out):
        orig_forward = module.forward
        twin = {"model": None}
        step = {"n": 0}

        def _cpu_twin():
            if twin["model"] is None:
                logger.info("[PCC] loading bf16 CPU twin: {}", name)
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
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed. Calculated: pcc=-0.084473 (transformer forward 17, "
        "step 9 conditional CFG pass). Required: pcc=0.9 "
        "https://github.com/tenstorrent/tt-xla/issues/5991"
    )
)
def test_pipeline():
    """Run the HunyuanVideo15 pipeline with per-component PCC vs CPU twins."""
    xr.set_device_type("TT")

    pipeline = HunyuanVideo15Pipeline(
        config=HunyuanVideo15Config(
            num_inference_steps=NUM_INFERENCE_STEPS, num_frames=NUM_FRAMES
        )
    )
    pipeline.setup()
    _attach_pcc_checks(pipeline)

    pipeline.generate(prompt=PROMPT, seed=SEED)
