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
must **agree** whether the prefill ran in a single chunk
(``max_num_batched_tokens`` >= prompt length) or in several chunks
(small ``max_num_batched_tokens``). The two paths use numerically different
attention kernels, so we compare the leading greedy tokens rather than the full
sequence (see ``_MATCH_PREFIX_TOKENS``): a cached-prefix attention bug diverges
from the first generated token, whereas low-precision argmax drift only flips a
late token. A leading-prefix mismatch means the chunk-N queries did not
correctly attend to the cached prefix (the cached-prefill attention bug).

Prefix caching is disabled in the equivalence test so the *only* variable is
the chunk size.
"""
import pytest
import vllm

_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
_MAX_MODEL_LEN = 2048
_MAX_TOKENS = 32
# Number of leading greedy tokens that must match between single-chunk and
# multi-chunk prefill. We compare a prefix rather than the full sequence
# because the two paths use numerically different attention kernels (single-shot
# SDPA vs the chunked-SDPA op) and different prefill bucket shapes, so at low
# precision they accumulate slightly differently and a late greedy argmax can
# flip -- benign drift, not an attention bug. (SDPA still lacks an fp32
# accumulation path: tt-mlir #8657.) A real cached-prefix bug corrupts attention
# from the first generated token, so a matching leading prefix is a strong guard.
# On device the shared prefix is ~17 tokens at bfp8 and ~30 at bf16+fp32-acc;
# 8 sits comfortably inside the low-precision margin.
_MATCH_PREFIX_TOKENS = 8

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
    out = llm.generate([_PROMPT], sp)[0].outputs[0]
    text, token_ids = out.text, list(out.token_ids)
    del llm
    return text, token_ids


@pytest.mark.nightly
@pytest.mark.single_device
def test_chunked_prefill_matches_single_chunk():
    """Multi-chunk prefill must reproduce single-chunk greedy generation.

    Oracle: budget >= prompt length -> one prefill chunk (num_computed == 0,
    standard attention path). Multi-chunk: small budget -> the prompt is split,
    so chunks 2..N exercise the cached-prefix on-device chunked-SDPA path.

    The two paths are numerically different (different attention kernels and
    prefill bucket shapes), so we assert the leading ``_MATCH_PREFIX_TOKENS``
    greedy tokens match -- a real cached-prefix attention bug diverges from the
    first generated token, while benign low-precision drift only flips a late
    argmax (see _MATCH_PREFIX_TOKENS; full-sequence equality is gated on SDPA
    fp32 accumulation, tt-mlir #8657).
    """
    single_text, single_ids = _generate(_MAX_MODEL_LEN)  # single chunk (oracle)
    multi_text, multi_ids = _generate(128)  # several block-aligned chunks

    print(f"single-chunk: {single_text!r}")
    print(f"multi-chunk:  {multi_text!r}")

    assert single_text, "single-chunk greedy generation was empty"
    assert multi_text, "multi-chunk greedy generation was empty"

    n = _MATCH_PREFIX_TOKENS
    assert len(single_ids) >= n and len(multi_ids) >= n, (
        f"need >= {n} generated tokens to compare "
        f"(single={len(single_ids)}, multi={len(multi_ids)})"
    )
    assert single_ids[:n] == multi_ids[:n], (
        f"first {n} greedy tokens differ between single-chunk and multi-chunk "
        "prefill -- cached-prefix attention is incorrect (a precision flip would "
        "occur later than this).\n"
        f"  single_ids[:{n}]={single_ids[:n]}\n  multi_ids[:{n}]={multi_ids[:n]}\n"
        f"  single={single_text!r}\n  multi={multi_text!r}"
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_chunk_budget_decoupled_from_max_model_len():
    """A budget smaller than max_model_len must be accepted (chunked prefill).

    Before #4986 this raised at model-runner init
    (max_model_len * max_num_seqs <= max_num_batched_tokens). It must now run.
    """
    text, _ = _generate(256)  # 256 << max_model_len (2048)
    assert text, "generation with budget << max_model_len was empty"


@pytest.mark.nightly
@pytest.mark.single_device
def test_chunked_prefill_batch_all_users_match(monkeypatch):
    """Batch>1 chunked prefill must produce identical output for all users.

    Guards against bugs that hide in the batch dimension: if only user 0 is
    correct (e.g. a stale page-table pointer or bad GQA broadcast), this test
    catches it. Also sets VLLM_XLA_CHECK_RECOMPILATION=1 to verify that the
    chunked-prefill path reuses compiled graphs after warm-up (no recompile).
    """
    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "1")

    chunk = 32
    n_users = 4
    max_tokens = 16
    prompt = "The quick brown fox jumps over the lazy dog and then runs away quickly"

    llm = vllm.LLM(
        model="facebook/opt-125m",
        max_model_len=256,
        max_num_seqs=n_users,
        max_num_batched_tokens=chunk * n_users,
        gpu_memory_utilization=0.001,
        enable_prefix_caching=False,
        additional_config={
            "prefill_chunk_size": chunk,
            "enable_const_eval": True,
            "min_context_len": 32,
        },
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    outputs = llm.generate([prompt] * n_users, sp)

    token_ids = [list(o.outputs[0].token_ids) for o in outputs]
    for i, ids in enumerate(token_ids):
        print(f"user {i}: {ids}")
        assert len(ids) == max_tokens, f"user {i} produced {len(ids)} tokens"

    for i in range(1, n_users):
        assert token_ids[i] == token_ids[0], (
            f"user {i} output differs from user 0 -- batch-dim attention bug.\n"
            f"  user 0: {token_ids[0]}\n  user {i}: {token_ids[i]}"
        )
