# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end check that ngram speculative decode does not change the output.

Two prompts of different lengths, so their decode positions sit on different KV
cache block boundaries. A speculative row is re-fed to its block boundary and the
prefix offset is one shared value per pass, so a batch spanning two boundaries has
to be split across passes; without that one request's K/V is written at the wrong
offset and its output drifts. Both runs use the same batch shape, leaving spec
decode as the only variable.

Repeated per parallelism mode, because the speculative row work touches page
tables and cache positions, which is exactly what TP shards by head and DP shards
by row (DP also pads the batch with zero-token rows).
"""

import gc

import pytest
import vllm

SINGLE_MODEL = "facebook/opt-125m"
# Multichip runs use the model the other TP/DP tests use.
MULTICHIP_MODEL = "Qwen/Qwen3-0.6B"

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


def _generate(
    model: str,
    speculative: bool,
    extra_config: dict | None = None,
    gpu_memory_utilization: float = 0.02,
) -> dict[str, list[int]]:
    additional_config = {"min_context_len": 32}
    additional_config.update(extra_config or {})
    llm_args = {
        "model": model,
        "max_num_seqs": len(PROMPTS),
        "max_model_len": 256,
        # Spec decode requires max_num_batched_tokens >= max_model_len x
        # max_num_seqs, so scale it with the batch.
        "max_num_batched_tokens": 256 * len(PROMPTS),
        "gpu_memory_utilization": gpu_memory_utilization,
        "additional_config": additional_config,
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


def _assert_spec_matches_greedy(mode: str, **kwargs) -> None:
    baseline = _generate(speculative=False, **kwargs)
    speculative = _generate(speculative=True, **kwargs)

    assert set(baseline) == set(PROMPTS)
    for prompt in PROMPTS:
        assert baseline[prompt], f"no tokens generated for {prompt!r}"
        assert speculative[prompt] == baseline[prompt], (
            f"greedy + ngram spec decode ({mode}) must be token identical\n"
            f"  prompt:      {prompt!r}\n"
            f"  baseline:    {baseline[prompt]}\n"
            f"  speculative: {speculative[prompt]}"
        )


@pytest.mark.push
@pytest.mark.single_device
def test_ngram_spec_decode_matches_greedy():
    """Single device. Covers one request and, in the same batch, rows whose decode
    positions sit on different KV block boundaries."""
    _assert_spec_matches_greedy("single device", model=SINGLE_MODEL)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.dual_chip
def test_ngram_spec_decode_matches_greedy_tensor_parallel():
    """Tensor parallel, where the KV cache is sharded by head."""
    _assert_spec_matches_greedy(
        "tensor parallel",
        model=MULTICHIP_MODEL,
        extra_config={"enable_tensor_parallel": True},
        gpu_memory_utilization=0.1,
    )


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.dual_chip
def test_ngram_spec_decode_matches_greedy_data_parallel():
    """Data parallel, where rows are sharded across chips and the batch is padded
    with zero-token rows, which is what the boundary trim has to cope with."""
    _assert_spec_matches_greedy(
        "data parallel",
        model=MULTICHIP_MODEL,
        extra_config={"enable_data_parallel": True},
        gpu_memory_utilization=0.1,
    )


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
def test_ngram_spec_decode_matches_greedy_data_tensor_parallel():
    """DP + TP together, one sentence per replica.

    len(PROMPTS) == dp_size keeps this at per-device batch 1, matching
    test_data_tensor_parallel_generation_push. Wide batch DP+TP has a separate
    open correctness bug (first local user of each replica), so going wider here
    would corrupt the baseline and make the comparison meaningless.

    Per-device batch 1 means each pass carries one row, so this does not exercise
    the boundary trim; it checks that spec decode works at all with the cache
    sharded on both axes.
    """
    _assert_spec_matches_greedy(
        "data + tensor parallel",
        model=MULTICHIP_MODEL,
        extra_config={"enable_tensor_parallel": True, "enable_data_parallel": True},
        gpu_memory_utilization=0.1,
    )
