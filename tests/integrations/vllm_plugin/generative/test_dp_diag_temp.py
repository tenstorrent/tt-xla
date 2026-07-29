# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEMPORARY dp=2 diagnostic -- DELETE BEFORE MERGE.

`test_data_parallel_chunked_prefill_n300` still corrupts replica 1 at dp=2 even
with the three DP fixes on this branch, while the equivalent config at dp=8 is
clean locally. dp=2 is not reproducible on an 8-chip llmbox (UMD discovers eth
links to invisible cards), so this runs the isolation matrix on an n300 in CI.

Each case prints per-row token ids and the first position where a row diverges
from row 0. Identical prompts + greedy means every row MUST emit the same
sequence, so divergence at position 0 means prefill produced bad logits.

Run with:
  gh workflow run manual-test-single.yml --ref <branch> \
    -f dir=tests/integrations/vllm_plugin/generative/test_dp_diag_temp.py \
    -f runs_on=n300 -f args="-s"
"""

import gc

import pytest
import vllm
from chunked_prefill_data import CHUNKED_PREFILL_PROMPT


def _run(*, n_seqs, chunk, prefix_cache, distinct, max_model_len=512):
    prompts = (
        [CHUNKED_PREFILL_PROMPT, CHUNKED_PREFILL_PROMPT[:-40]][:n_seqs]
        if distinct
        else [CHUNKED_PREFILL_PROMPT] * n_seqs
    )
    while len(prompts) < n_seqs:  # pad out for n_seqs > 2
        prompts.append(prompts[-1] + " Also note this.")
    llm = vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_num_seqs=n_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=prefix_cache,
        additional_config={
            "min_context_len": 128,
            "enable_data_parallel": True,
            "prefill_chunk_size": chunk,
        },
    )
    try:
        outs = llm.generate(
            prompts, vllm.SamplingParams(temperature=0.0, max_tokens=16)
        )
        ids = [list(o.outputs[0].token_ids) for o in outs]
        texts = [o.outputs[0].text for o in outs]
    finally:
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del llm
        gc.collect()
    return ids, texts


def _report(tag, ids, texts, *, expect_equal):
    print(f"\n===== {tag} =====", flush=True)
    ref = ids[0]
    for i, (row, txt) in enumerate(zip(ids, texts)):
        diff = next(
            (k for k in range(min(len(ref), len(row))) if ref[k] != row[k]), None
        )
        print(
            f"  row {i}: first_diff={diff} ids[:6]={row[:6]} {txt[:48]!r}",
            flush=True,
        )
    if expect_equal:
        mismatched = [i for i, row in enumerate(ids) if row != ref]
        print(f"  MISMATCHED ROWS: {mismatched}", flush=True)


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.dual_chip
def test_dp2_multichunk_identical():
    """The failing config: 2 identical prompts, 2 chunks, dp=2."""
    ids, texts = _run(n_seqs=2, chunk=128, prefix_cache=True, distinct=False)
    _report(
        "dp2 multichunk identical (the failing case)", ids, texts, expect_equal=True
    )


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.dual_chip
def test_dp2_singlechunk_identical():
    """Same but one chunk -- isolates the continuation path at dp=2."""
    ids, texts = _run(n_seqs=2, chunk=512, prefix_cache=True, distinct=False)
    _report("dp2 singlechunk identical", ids, texts, expect_equal=True)


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.dual_chip
def test_dp2_multichunk_distinct():
    """Distinct prompts -- rows need not match, but each must be coherent."""
    ids, texts = _run(n_seqs=2, chunk=128, prefix_cache=True, distinct=True)
    _report("dp2 multichunk distinct", ids, texts, expect_equal=False)


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.dual_chip
def test_dp2_multichunk_identical_wide():
    """max_num_seqs=4 at dp=2 -> 2 rows per replica (local_batch=2)."""
    ids, texts = _run(n_seqs=4, chunk=128, prefix_cache=True, distinct=False)
    _report("dp2 multichunk identical, local_batch=2", ids, texts, expect_equal=True)
