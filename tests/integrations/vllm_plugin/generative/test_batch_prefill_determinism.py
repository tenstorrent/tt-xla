# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test for batched-prefill non-determinism (tt-xla #<TBD>).

Greedy (temperature=0) decoding of N *identical* prompts in one batch must
produce N *identical* outputs: each row is an independent computation, so
argmax decoding cannot legitimately differ between rows.

On the TT backend, with ``fp32_dest_acc_en=False`` (low-precision / bf16 matmul
destination accumulation — the default used by the perf benchmarks), batched
prefill becomes batch-position-dependent: at batch sizes >= 4 the per-row
logits differ slightly, flipping near-tie greedy tokens, so identical prompts
diverge. With fp32 destination accumulation (``fp32_dest_acc_en=True`` or the
tt-mlir default) the rows agree.

It is latent in the benchmark suite because those tests only assert token
*counts* / User-0 PCC, not output consistency across users, and the default
benchmark config sets ``fp32_dest_acc_en=False`` with ``batch_size=32`` — which
produces several distinct outputs across the 32 identical prompts.
"""
import pytest
import vllm

_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
_BATCH = 4  # batch=2 does NOT trigger; >=4 does.
_MAX_TOKENS = 16

# Repetitive prompt: maximizes near-tie continuations, making the precision
# divergence observable. (Strongly-determined prompts may not flip any token.)
_PARA = (
    "The history of computing spans many centuries. Early mechanical "
    "calculators gave way to electromechanical machines, and then to the "
    "electronic digital computers that define the modern era. Each generation "
    "brought dramatic improvements in speed, reliability, and cost. "
)
_PROMPT = (
    "Summarize the following text in one sentence.\n\n" + (_PARA * 12) + "\nSummary:"
)


def _generate_batch(fp32_dest_acc_en: bool) -> list[str]:
    llm = vllm.LLM(
        model=_MODEL,
        max_model_len=2048,
        max_num_seqs=_BATCH,
        max_num_batched_tokens=_BATCH * 2048,  # whole batch prefilled in one step
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
        additional_config={
            "experimental_weight_dtype": "bfp_bf8",
            "experimental_kv_cache_dtype": "bfp_bf8",
            "fp32_dest_acc_en": fp32_dest_acc_en,
            "enable_trace": False,
        },
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS, ignore_eos=True)
    outs = llm.generate([_PROMPT] * _BATCH, sp)
    texts = [o.outputs[0].text for o in outs]
    del llm
    return texts


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Batched-prefill non-determinism with fp32_dest_acc_en=False: identical "
        "greedy prompts diverge at batch>=4 (low-precision matmul dest "
        "accumulation is batch-position-dependent). Remove xfail when the "
        "kernel-level fix lands (or the platform forces fp32 accumulation)."
    ),
)
def test_batch_prefill_identical_prompts_low_precision_accum():
    texts = _generate_batch(fp32_dest_acc_en=False)
    assert len(set(texts)) == 1, (
        "identical greedy prompts produced different outputs in one batch "
        f"({len(set(texts))} distinct):\n"
        + "\n".join(f"  [{i}] {t!r}" for i, t in enumerate(texts))
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_batch_prefill_identical_prompts_fp32_accum():
    """Control / fix verification: with fp32 destination accumulation the same
    batch is deterministic."""
    texts = _generate_batch(fp32_dest_acc_en=True)
    assert len(set(texts)) == 1, (
        "identical greedy prompts produced different outputs even with "
        f"fp32_dest_acc_en=True ({len(set(texts))} distinct):\n"
        + "\n".join(f"  [{i}] {t!r}" for i, t in enumerate(texts))
    )
