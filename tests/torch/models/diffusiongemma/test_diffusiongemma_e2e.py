# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DiffusionGemma 26B -- nightly PCC-gated block-diffusion e2e test on Tenstorrent.

Runs the shared ``tt_forge_models`` pipeline, the same code the demo and the
benchmark use; this module only adds PCC gating through the pipeline's ``_check``
seam. Two components are gated: the encoder prefill, once per canvas block, and
the decoder forward, once per step.

The trajectory is advanced with the *device* output, so a PCC drop fails the test
rather than silently degrading the text.
"""

import inspect
import os

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from utils import BringupStatus, Category, ModelGroup

from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.diffusiongemma.pytorch import (
    loader as diffgemma_loader,
)

MAX_NEW_TOKENS = 256
SEED = 0
# Measured worst case across the four input cases, rounded down. The encoder floor
# is the lower of the two because its error is accumulation over 57 layers, at a
# rate set by sequence length -- tenstorrent/tt-xla#6054. Decoder steps are
# measured in isolation, so theirs is per-step error, not cumulative.
ENCODER_PCC_THRESHOLD = 0.88
DECODER_PCC_THRESHOLD = 0.94

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _record_properties(model_name):
    return pytest.mark.record_test_properties(
        category=Category.MODEL_TEST,
        model_name=model_name,
        model_group=ModelGroup.GENERALITY,
        run_mode=RunMode.INFERENCE,
        bringup_status=BringupStatus.PASSED,
    )


# Log PCC instead of asserting it, so a run can verify staging/OOM without a
# floor aborting it at the encoder.
_PCC_SOFT = os.environ.get("DIFFGEMMA_PCC_SOFT") == "1"
# Skip the golden entirely: PCC_SOFT still runs a CPU forward of the 26B model
# on every step, which dominates a run that is not about numerics.
_PCC_OFF = os.environ.get("DIFFGEMMA_PCC_OFF") == "1"
# Denoise steps that get a CPU twin: one 26B CPU forward dominates each step.
PCC_CHECK_STEPS = int(os.environ.get("DIFFGEMMA_PCC_CHECK_STEPS", "10"))
# Token count for the text_long case -- matches the image cases (277-284).
TEXT_LONG_TOKENS = int(os.environ.get("DIFFGEMMA_TEXT_LONG_TOKENS", "277"))


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.llmbox
@pytest.mark.parametrize(
    "modality",
    [
        pytest.param("text", marks=_record_properties("DiffusionGemma_e2e")),
        pytest.param(
            "text_long", marks=_record_properties("DiffusionGemma_e2e_text_long")
        ),
        pytest.param("image", marks=_record_properties("DiffusionGemma_e2e_image")),
        pytest.param(
            "image_only",
            marks=_record_properties("DiffusionGemma_e2e_image_only"),
        ),
    ],
)
def test_diffusiongemma_e2e(modality):
    """Staged both-on-TT block diffusion with per-component PCC checks."""
    # transformers>=5.11 is required for DiffusionGemma; install it from the loader's
    # requirements.txt for this test only, roll back on exit (env stays clean for others).
    loader_path = inspect.getsourcefile(diffgemma_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import (
            DiffusionGemmaConfig,
            DiffusionGemmaPipeline,
        )

        class PccDiffusionGemmaPipeline(DiffusionGemmaPipeline):
            """The shipped pipeline with a PCC check on every component forward."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.records = []
                self._step = 0

            def _check(self, name, tt_out, golden_fn):
                if name == "encoder":
                    self._step = 0
                    label = name
                else:
                    self._step += 1
                    label = f"{name} step={self._step}"
                if _PCC_OFF:
                    self.records.append((name, self._step, 1.0))
                    return
                # The golden is a CPU forward of the 26B twin and dominates the
                # step time, so gate the leading denoise steps -- where a
                # numerical break shows up first -- instead of every step. Same
                # approach as tests/torch/models/flux (PCC_CHECK_STEPS there).
                if name != "encoder" and self._step > PCC_CHECK_STEPS:
                    return
                golden = golden_fn()
                reference = (
                    golden.last_hidden_state if name == "encoder" else golden.logits
                )
                pcc = _pcc(tt_out, reference)
                self.records.append((name, self._step, pcc))
                logger.info("[PCC] {}: pcc={:.6f}", label, pcc)
                floor = (
                    ENCODER_PCC_THRESHOLD
                    if name == "encoder"
                    else DECODER_PCC_THRESHOLD
                )
                if not (_PCC_SOFT or _PCC_OFF):
                    assert (
                        pcc >= floor
                    ), f"{label} PCC {pcc:.6f} below threshold {floor}"

        xr.set_device_type("TT")
        torch.manual_seed(SEED)

        pipeline = PccDiffusionGemmaPipeline(
            config=DiffusionGemmaConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                seed=SEED,
                image=modality in ("image", "image_only"),
            )
        )
        pipeline.setup()
        # image_only drops the text turn: the image is the whole prompt. text_long
        # runs the text path at the image cases' token count so the two are
        # comparable like for like (the short text case is only 15 tokens).
        if modality == "image_only":
            prompt = ""
        elif modality == "text_long":
            prompt, n = pipeline.loader.build_prompt(TEXT_LONG_TOKENS)
            logger.info("[text_long] prompt is {} tokens", n)
        else:
            prompt = os.environ.get("DIFFGEMMA_TEXT_PROMPT") or None
        text_out = pipeline.generate(prompt=prompt)
        logger.info("[{}] generated:\n{}", modality, text_out)

        # Guard against a vacuous pass: with no records `worst` would fall back to its
        # default and the assert below would succeed without a single check having run.
        assert (
            pipeline.records
        ), "no PCC checks ran: encoder/decoder forwards never fired"
        worst_enc = min(
            (p for n, _, p in pipeline.records if n == "encoder"), default=1.0
        )
        worst_dec = min(
            (p for n, _, p in pipeline.records if n != "encoder"), default=1.0
        )
        logger.info(
            "[{}] per-iteration PCC: {} checks, encoder={:.6f} worst decoder={:.6f}",
            modality,
            len(pipeline.records),
            worst_enc,
            worst_dec,
        )
        if not (_PCC_SOFT or _PCC_OFF):
            # separate floors: the encoder is accumulation-limited, the decoder is not
            assert worst_enc >= ENCODER_PCC_THRESHOLD
            assert worst_dec >= DECODER_PCC_THRESHOLD
