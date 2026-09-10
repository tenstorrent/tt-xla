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
PCC_THRESHOLD = 0.96

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


# The image path currently measures ~0.87 (sequence-length decay plus the vision
# front-end's seed error, tt-xla#6054). Set DIFFGEMMA_PCC_SOFT=1 to log PCC instead
# of asserting, so a run can be used to verify staging/OOM without the floor
# aborting it at the encoder. The committed floor itself is not lowered.
_PCC_SOFT = os.environ.get("DIFFGEMMA_PCC_SOFT") == "1"
# DIFFGEMMA_PCC_OFF skips the golden ENTIRELY. PCC_SOFT still runs golden_fn()
# -- a CPU forward of the 26B model on every denoising step -- so it is useless
# for questions that are not about numerics (e.g. proving the staged residency
# no longer OOMs). OFF exercises the real TT pipeline and nothing else.
_PCC_OFF = os.environ.get("DIFFGEMMA_PCC_OFF") == "1"


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.llmbox
@pytest.mark.parametrize(
    "modality",
    [
        pytest.param("text", marks=_record_properties("DiffusionGemma_e2e")),
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
                if _PCC_OFF:
                    self.records.append((name, self._step, 1.0))
                    return
                golden = golden_fn()
                if name == "encoder":
                    reference = golden.last_hidden_state
                    self._step = 0
                    label = name
                else:
                    reference = golden.logits
                    self._step += 1
                    label = f"{name} step={self._step}"
                pcc = _pcc(tt_out, reference)
                self.records.append((name, self._step, pcc))
                logger.info("[PCC] {}: pcc={:.6f}", label, pcc)
                if not (_PCC_SOFT or _PCC_OFF):
                    assert (
                        pcc >= PCC_THRESHOLD
                    ), f"{label} PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"

        xr.set_device_type("TT")
        torch.manual_seed(SEED)

        pipeline = PccDiffusionGemmaPipeline(
            config=DiffusionGemmaConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                seed=SEED,
                image=(modality != "text"),
            )
        )
        pipeline.setup()
        # image_only drops the text turn: the image is the whole prompt.
        text_out = pipeline.generate(
            prompt="" if modality == "image_only" else None
        )
        logger.info("[{}] generated:\n{}", modality, text_out)

        # Guard against a vacuous pass: with no records `worst` would fall back to its
        # default and the assert below would succeed without a single check having run.
        assert (
            pipeline.records
        ), "no PCC checks ran: encoder/decoder forwards never fired"
        worst = min(p for *_, p in pipeline.records)
        logger.info(
            "[{}] per-iteration PCC: {} checks, worst={:.6f}",
            modality,
            len(pipeline.records),
            worst,
        )
        if not (_PCC_SOFT or _PCC_OFF):
            assert worst >= PCC_THRESHOLD
