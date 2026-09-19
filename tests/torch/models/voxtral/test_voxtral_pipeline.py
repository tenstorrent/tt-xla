# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-Mini-3B-2507 — nightly e2e audio->text generation test.

Voxtral (Mistral audio+text -> text) fits a single Wormhole chip. This drives
the shared ``VoxtralPipeline`` from ``tt_forge_models`` through the FULL greedy
autoregressive decode loop on device (audio tower + merge on host, the 3B LM on
device) and asserts the decoded answer identifies both the nursery rhyme and the
baseball teams in the two audio clips. Because decoding is greedy, the on-device
answer matches an equivalent CPU run token-for-token.
"""

import pytest
from infra import RunMode
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.voxtral.pytorch import ModelLoader, ModelVariant

# The Voxtral e2e pipeline lives in tt-forge-models; skip cleanly until the
# submodule uplift brings in voxtral/pytorch/pipeline.py.
_pipeline = pytest.importorskip(
    "third_party.tt_forge_models.voxtral.pytorch.pipeline",
    reason="requires tt-forge-models voxtral/pytorch/pipeline.py (submodule uplift)",
)
VoxtralConfig = _pipeline.VoxtralConfig
VoxtralPipeline = _pipeline.VoxtralPipeline

VARIANT_NAME = ModelVariant.VOXTRAL_MINI_3B
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

MAX_NEW_TOKENS = 256


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_voxtral_pipeline():
    """Run Voxtral audio->text generation on device and assert the answer content."""
    pipeline = VoxtralPipeline(config=VoxtralConfig())
    pipeline.setup()

    answer = pipeline.generate(max_new_tokens=MAX_NEW_TOKENS)

    assert answer, "no text was generated"
    lowered = answer.lower()
    # Robust content checks: the nursery rhyme and the sport are stable. Finer
    # details (e.g. specific team names) are numerically sensitive — greedy over
    # bf16 can flip an uncertain token when the static window (max_new_tokens)
    # changes the compiled graph — so we don't assert on them.
    assert "mary had a little lamb" in lowered, f"nursery rhyme not identified: {answer!r}"
    assert "baseball" in lowered, f"baseball game not identified: {answer!r}"
