# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end check that ngram speculative decode does not change the output.

A draft token is only accepted when it matches what the target model would have
sampled anyway, so with greedy sampling spec decode must be token identical to
plain greedy. That is a correctness oracle needing no golden text: the non
speculative run is the reference, so the check survives model, dtype and compiler
changes.

The unit tests in sampling/test_speculative_decode.py cover the rejection sampler
and the proposer handoff on cpu with fakes. Nothing there runs a model, and
assert_output_coherent only detects token soup, so several device level defects in
the speculative path emitted fluent but wrong text unnoticed.

Two prompts of different lengths, so their decode positions sit on different KV
cache block boundaries. A speculative row is re-fed to its block boundary and the
prefix offset is one shared value per pass, so a batch spanning two boundaries has
to be split across passes; without that one request's K/V is written at the wrong
offset and its output drifts. Both runs use the same batch shape, leaving spec
decode as the only variable.
"""

import gc

import pytest
import vllm

MODEL = "facebook/opt-125m"

# Repetitive text so the ngram proposer finds matches and actually drafts.
_UNIT = "The cat sat on the mat. The dog sat on the log. "
LONG_PROMPT = _UNIT * 3 + "The cat sat on the"
SHORT_PROMPT = "The cat sat on the mat. The cat sat on the"
PROMPTS = [LONG_PROMPT, SHORT_PROMPT]

NUM_SPEC_TOKENS = 3
MAX_TOKENS = 32


def _shutdown(llm: vllm.LLM) -> None:
    """Terminate the EngineCore subprocess before the next engine is built.

    Dropping the reference is not enough: the subprocess keeps holding
    /dev/tenstorrent and the next vllm.LLM() hangs. Same reason
    sampling/conftest.py shuts engines down explicitly between modules.
    """
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    gc.collect()


def _generate(speculative: bool) -> dict[str, list[int]]:
    llm_args = {
        "model": MODEL,
        "max_num_seqs": len(PROMPTS),
        "max_model_len": 256,
        # Spec decode requires max_num_batched_tokens >= max_model_len x
        # max_num_seqs, so scale it with the batch.
        "max_num_batched_tokens": 256 * len(PROMPTS),
        "gpu_memory_utilization": 0.02,
        "additional_config": {"min_context_len": 32},
    }
    if speculative:
        llm_args["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        }

    llm = vllm.LLM(**llm_args)
    try:
        outs = llm.generate(
            PROMPTS, vllm.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        )
        # generate() may reorder, so key by prompt rather than position.
        return {o.prompt: list(o.outputs[0].token_ids) for o in outs}
    finally:
        _shutdown(llm)


@pytest.mark.push
@pytest.mark.single_device
def test_ngram_spec_decode_matches_greedy():
    """Greedy plus spec decode must be token identical to plain greedy.

    Covers a single request (the long prompt) and, in the same batch, rows whose
    decode positions sit on different KV block boundaries.
    """
    baseline = _generate(speculative=False)
    speculative = _generate(speculative=True)

    assert set(baseline) == set(PROMPTS)
    for prompt in PROMPTS:
        assert baseline[prompt], f"no tokens generated for {prompt!r}"
        assert speculative[prompt] == baseline[prompt], (
            "greedy + ngram spec decode must be token identical to greedy\n"
            f"  prompt:      {prompt!r}\n"
            f"  baseline:    {baseline[prompt]}\n"
            f"  speculative: {speculative[prompt]}"
        )
