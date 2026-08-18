# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-Mini-3B-2507 — nightly end-to-end audio+text -> text pipeline test.

Runs the full audio-understanding pipeline: the Whisper-style audio tower and the
audio/text merge run on the host (the HF merge is a data-dependent
``masked_scatter`` that cannot be lowered), and the 30-layer Ministral-3B
language model greedy-decodes the answer on Tenstorrent. The pipeline
implementation lives in ``tt_forge_models`` and is shared with the audio
benchmark (``tests/benchmark/test_audio.py``).

The test asserts the model actually answered the sample question — the answer has
to name the sport in the audio, a word that appears nowhere in the prompt —
instead of only checking that some text came back.
"""

from pathlib import Path

import pytest
import torch_xla.runtime as xr
from infra import RunMode
from loguru import logger
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.voxtral.pytorch.pipeline import (
    VoxtralConfig,
    VoxtralPipeline,
)

# Token budget for the answer. The decode loop carries no KV cache, so every
# step is a full-window forward, which bounds how long this can run: 48 tokens
# is a full answer to the sample question plus margin, so the keyword assertion
# below does not depend on the model naming both clips in its first few words.
MAX_NEW_TOKENS = 48

OUTPUT_PATH = "voxtral_mini_3b_answer.txt"

# The sample conversation feeds two clips (a baseball radio call and "Mary Had a
# Little Lamb") and asks which sport and which nursery rhyme are referenced.
# "baseball" is the check that matters: the word is nowhere in the prompt, so the
# model can only produce it by understanding the audio. Which *game* it names, and
# whether it gets to naming the rhyme within the token budget, vary with device
# numerics (n150 and p150 word the answer differently), so neither is asserted.
EXPECTED_KEYWORD = "baseball"

# A one-word reply would satisfy the keyword check without being an answer.
MIN_ANSWER_WORDS = 8


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="Voxtral_Mini_3B_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_voxtral_mini_3b_pipeline():
    """Full Voxtral audio->text chain with the language model on TT."""
    xr.set_device_type("TT")

    output_file = Path(OUTPUT_PATH)
    if output_file.exists():
        output_file.unlink()

    pipeline = VoxtralPipeline(config=VoxtralConfig(max_new_tokens=MAX_NEW_TOKENS))
    pipeline.setup()
    answer = pipeline.generate(max_new_tokens=MAX_NEW_TOKENS)

    output_file.write_text(answer)
    logger.info(f"Voxtral answer ({len(answer)} chars): {answer!r}")

    steps = pipeline._perf["steps"]
    assert steps, "Decode loop ran no steps"

    assert len(answer.split()) >= MIN_ANSWER_WORDS, (
        f"answer is too short to be an answer ({len(answer.split())} words < "
        f"{MIN_ANSWER_WORDS}): {answer!r}"
    )
    assert EXPECTED_KEYWORD in answer.lower(), (
        f"answer does not name the sport in the audio ({EXPECTED_KEYWORD!r}) — the "
        f"model did not understand the audio prompt. Answer: {answer!r}"
    )

    logger.info(
        "[voxtral] prefill_tokens={} decode_steps={} answer saved to {}",
        pipeline._perf["prefill_tokens"],
        len(steps),
        OUTPUT_PATH,
    )
