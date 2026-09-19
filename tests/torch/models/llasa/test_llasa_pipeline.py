# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llasa-8B (HKUSTAudio/Llasa-8B) — nightly e2e text-to-speech generation test.

Llasa-8B is a ~16 GB Llama-3.1-8B TTS fine-tune whose weights overflow a single
Wormhole chip, so the LM runs **tensor-parallel** across a multi-chip mesh
(Megatron-1D, lm_head replicated). This drives the shared ``LlasaPipeline`` from
``tt_forge_models`` through the FULL autoregressive decode loop on device and
asserts it emits valid XCodec2 speech-token codes. Turning codes into a waveform
is XCodec2's job (a separate torch-2.5 env) and is intentionally out of scope.
"""

import pytest
from infra import RunMode
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.llasa.causal_lm.pytorch import ModelLoader, ModelVariant

# The Llasa e2e pipeline lives in tt-forge-models; skip cleanly until the
# submodule uplift brings in llasa/causal_lm/pytorch/pipeline.py.
_pipeline = pytest.importorskip(
    "third_party.tt_forge_models.llasa.causal_lm.pytorch.pipeline",
    reason="requires tt-forge-models llasa/causal_lm/pytorch/pipeline.py (submodule uplift)",
)
LlasaConfig = _pipeline.LlasaConfig
LlasaPipeline = _pipeline.LlasaPipeline

VARIANT_NAME = ModelVariant.LLASA_8B
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

TEXT = "Tenstorrent builds hardware and software for AI."
MAX_NEW_TOKENS = 400
SEED = 42


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_llasa_pipeline():
    """Run Llasa-8B TTS generation (LM tensor-parallel) and assert speech codes."""
    pipeline = LlasaPipeline(config=LlasaConfig())
    pipeline.setup()

    codes = pipeline.generate(text=TEXT, max_new_tokens=MAX_NEW_TOKENS, seed=SEED)

    assert len(codes) > 0, "no speech tokens were generated"
    assert all(0 <= c <= 65535 for c in codes), "speech codes out of XCodec2 range"
    # A well-formed utterance stops on <|SPEECH_GENERATION_END|> before the cap.
    assert len(codes) < MAX_NEW_TOKENS, "generation hit the cap without stopping"
