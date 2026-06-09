# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill correctness + decoupling tests (tt-xla #4986).

The TT backend caps the per-step prefill token budget
(``max_num_batched_tokens``) independently of ``max_model_len`` so that
precompile buckets and prefill activation memory stay bounded by a small chunk
size. A prompt longer than the budget is processed in multiple prefill chunks
(see ``AscendScheduler`` + the cached-prefix attention path in ``attention.py``
``_compute_full_attention``).

The key correctness invariant: greedy (temperature=0) generation of a prompt
must be **identical** whether the prefill ran in a single chunk
(``max_num_batched_tokens`` >= prompt length) or in several chunks
(small ``max_num_batched_tokens``). A mismatch means the chunk-N queries did
not correctly attend to the cached prefix (the cached-prefill attention bug).

Prefix caching is disabled in the equivalence test so the *only* variable is
the chunk size.
"""
import pytest
import vllm

_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
_MAX_MODEL_LEN = 2048
_MAX_TOKENS = 32

# A prompt comfortably longer than the small chunk budget so it must be split
# into several prefill chunks.
_PARA = (
    "The history of computing spans many centuries. Early mechanical "
    "calculators gave way to electromechanical machines, and then to the "
    "electronic digital computers that define the modern era. Each generation "
    "brought dramatic improvements in speed, reliability, and cost. "
)
_PROMPT = (
    "Summarize the following text in one sentence.\n\n" + (_PARA * 12) + "\nSummary:"
)


def _generate(max_num_batched_tokens: int) -> str:
    llm = vllm.LLM(
        model=_MODEL,
        max_model_len=_MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
        additional_config={
            "experimental_weight_dtype": "bfp_bf8",
            "experimental_kv_cache_dtype": "bfp_bf8",
            "fp32_dest_acc_en": False,
            "enable_trace": False,
            # Opt in to chunked prefill at this chunk size. With budget >= prompt
            # length this is a single chunk (the oracle); with a small budget the
            # prompt is split into block-aligned chunks.
            "prefill_chunk_size": max_num_batched_tokens,
        },
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS, ignore_eos=True)
    text = llm.generate([_PROMPT], sp)[0].outputs[0].text
    del llm
    return text


@pytest.mark.nightly
@pytest.mark.single_device
def test_chunked_prefill_matches_single_chunk():
    """Greedy output must be identical with single-chunk vs multi-chunk prefill.

    Oracle: budget >= prompt length -> one prefill chunk (num_computed == 0,
    fast attention path). Multi-chunk: small budget -> the prompt is split, so
    chunks 2..N exercise the cached-prefix masked-SDPA attention path.
    """
    single = _generate(_MAX_MODEL_LEN)  # >= prompt length: single chunk
    multi = _generate(128)  # small budget: several block-aligned chunks

    print(f"single-chunk: {single!r}")
    print(f"multi-chunk:  {multi!r}")

    assert single, "single-chunk greedy generation was empty"
    assert multi, "multi-chunk greedy generation was empty"
    assert single == multi, (
        "greedy output differs between single-chunk and multi-chunk prefill — "
        "cached-prefix prefill attention is incorrect.\n"
        f"  single={single!r}\n  multi={multi!r}"
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_chunk_budget_decoupled_from_max_model_len():
    """A budget smaller than max_model_len must be accepted (chunked prefill).

    Before #4986 this raised at model-runner init
    (max_model_len * max_num_seqs <= max_num_batched_tokens). It must now run.
    """
    out = _generate(256)  # 256 << max_model_len (2048)
    assert out, "generation with budget << max_model_len was empty"
