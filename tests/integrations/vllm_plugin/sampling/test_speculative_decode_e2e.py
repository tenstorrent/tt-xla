# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end check that ngram speculative decode does not change the output.

A draft token is only accepted when it matches what the target model would have
sampled anyway, so with greedy sampling spec decode must be token-identical to
plain greedy. That makes a cheap correctness oracle needing no golden text: the
non-speculative run is the reference.

The unit tests in test_speculative_decode.py cover the rejection sampler and the
proposer handoff on cpu with fakes. Nothing there runs a model, which is why
several device-level defects in the speculative path went unnoticed.

Each engine config runs in a fresh subprocess: two ``vllm.LLM`` instances in one
process leave the first ``EngineCore`` holding /dev/tenstorrent and the second
stalls, so this module re-invokes itself as a worker (see ``__main__``).
"""

import json
import os
import subprocess
import sys

import pytest

MODEL = "facebook/opt-125m"

# Repetitive text so the ngram proposer finds matches and actually drafts.
_UNIT = "The cat sat on the mat. The dog sat on the log. "
LONG_PROMPT = _UNIT * 3 + "The cat sat on the"
# Deliberately a different length, so this request's decode position sits on a
# different KV-cache block boundary than LONG_PROMPT's. One shared prefix offset
# per pass means the runner must split such a batch by boundary.
SHORT_PROMPT = "The cat sat on the mat. The cat sat on the"

NUM_SPEC_TOKENS = 3
MAX_TOKENS = 32


def _run(prompts, speculative, max_num_seqs):
    """Run one engine config in a fresh process; return per-prompt token ids."""
    payload = json.dumps(
        {
            "prompts": prompts,
            "speculative": speculative,
            "max_num_seqs": max_num_seqs,
        }
    )
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), payload],
        capture_output=True,
        text=True,
        timeout=2400,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(
        f"worker produced no RESULT (exit {proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-2000:]}"
    )


@pytest.mark.push
@pytest.mark.single_device
def test_ngram_spec_decode_matches_greedy():
    """Single request: speculative output must equal plain greedy exactly."""
    prompts = [LONG_PROMPT]
    baseline = _run(prompts, speculative=False, max_num_seqs=1)
    speculative = _run(prompts, speculative=True, max_num_seqs=1)

    assert speculative == baseline, (
        "greedy + ngram spec decode must be token-identical to greedy\n"
        f"  baseline:    {baseline}\n"
        f"  speculative: {speculative}"
    )


@pytest.mark.push
@pytest.mark.single_device
def test_ngram_spec_decode_matches_greedy_mixed_block_boundaries():
    """Two requests whose decode positions sit on different KV block boundaries.

    A speculative row is re-fed to its block boundary so the block-quantised
    cache write lines up with the exact read offset, and the prefix offset is one
    shared value per pass. A batch spanning two boundaries therefore has to be
    split across passes; if it is not, one request's K/V is written at the wrong
    offset and its output silently drifts. Each request is compared against its
    own single-request greedy baseline.
    """
    prompts = [LONG_PROMPT, SHORT_PROMPT]
    baseline = [
        _run([LONG_PROMPT], speculative=False, max_num_seqs=1)[0],
        _run([SHORT_PROMPT], speculative=False, max_num_seqs=1)[0],
    ]
    speculative = _run(prompts, speculative=True, max_num_seqs=2)

    assert len(speculative) == 2
    for i, (got, want) in enumerate(zip(speculative, baseline)):
        assert got == want, (
            f"request {i}: greedy + spec decode must be token-identical\n"
            f"  baseline:    {want}\n"
            f"  speculative: {got}"
        )


def _worker(payload):
    import vllm

    cfg = json.loads(payload)
    max_num_seqs = cfg["max_num_seqs"]
    llm_args = {
        "model": MODEL,
        "max_num_seqs": max_num_seqs,
        "max_model_len": 256,
        # Spec decode requires max_num_batched_tokens >= max_model_len x
        # max_num_seqs, so scale it with the batch.
        "max_num_batched_tokens": 256 * max_num_seqs,
        "gpu_memory_utilization": 0.02,
        "additional_config": {"min_context_len": 32},
    }
    if cfg["speculative"]:
        llm_args["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        }

    llm = vllm.LLM(**llm_args)
    outs = llm.generate(
        cfg["prompts"], vllm.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    )
    # generate() may reorder; key by prompt to restore the caller's order.
    by_prompt = {o.prompt: list(o.outputs[0].token_ids) for o in outs}
    print("RESULT " + json.dumps([by_prompt[p] for p in cfg["prompts"]]), flush=True)


if __name__ == "__main__":
    _worker(sys.argv[1])
