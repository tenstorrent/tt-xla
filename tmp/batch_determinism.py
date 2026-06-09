# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal repro for batched-prefill non-determinism.

Run N IDENTICAL prompts in one batch with greedy (temperature=0) sampling.
Correct behavior: all N outputs identical (greedy + identical input => identical
logits per row => identical argmax). A divergence indicates a batched-prefill
(or batched-sampler) correctness bug.

Knobs (env):
  BD_BATCH         (default 4)     number of identical prompts
  BD_MML           (default 256)   max_model_len
  BD_CPU_SAMPLING  (default 0)     1 -> host sampler, 0 -> device tt::sampling
  BD_NLAYERS       (default 0)     override num_hidden_layers (0 = full model)
  BD_MAX_TOKENS    (default 16)
  BD_KV_DTYPE      (default "")    e.g. bfp_bf8
  BD_WEIGHT_DTYPE  (default "")    e.g. bfp_bf8
"""
import os

import vllm

BATCH = int(os.environ.get("BD_BATCH", "4"))
MML = int(os.environ.get("BD_MML", "256"))
CPU_SAMPLING = os.environ.get("BD_CPU_SAMPLING", "0") == "1"
NLAYERS = int(os.environ.get("BD_NLAYERS", "0"))
MAX_TOKENS = int(os.environ.get("BD_MAX_TOKENS", "16"))
KV_DTYPE = os.environ.get("BD_KV_DTYPE", "")
WEIGHT_DTYPE = os.environ.get("BD_WEIGHT_DTYPE", "")

_SHORT = (
    "Question: What is the capital of France, and why has it historically been "
    "an important center of art, politics, and trade in Europe? Answer:"
)
# Long, repetitive prompt (matches the condition where divergence was first seen).
_PARA = (
    "The history of computing spans many centuries. Early mechanical "
    "calculators gave way to electromechanical machines, and then to the "
    "electronic digital computers that define the modern era. Each generation "
    "brought dramatic improvements in speed, reliability, and cost. "
)
# The exact prompt the b32 benchmark uses (known to produce 3 distinct outputs).
_BENCH = "Here is an exhaustive list of the best practices for writing clean code:"
if os.environ.get("BD_BENCH_PROMPT", "0") == "1":
    PROMPT = _BENCH
elif os.environ.get("BD_LONG", "0") == "1":
    PROMPT = (
        "Summarize the following text in one sentence.\n\n"
        + (_PARA * 12)
        + "\nSummary:"
    )
else:
    PROMPT = _SHORT


def main():
    const_eval = os.environ.get("BD_CONST_EVAL", "0") == "1"
    additional = {"enable_const_eval": const_eval, "min_context_len": 32}
    # BD_FP32_ACC: "0" -> fp32_dest_acc_en=False (low-precision matmul accum),
    # "1" -> True, unset -> leave at tt-mlir default.
    if "BD_FP32_ACC" in os.environ:
        additional["fp32_dest_acc_en"] = os.environ["BD_FP32_ACC"] == "1"
    if CPU_SAMPLING:
        additional["cpu_sampling"] = True
    if NLAYERS > 0:
        additional["num_hidden_layers"] = NLAYERS
    if KV_DTYPE:
        additional["experimental_kv_cache_dtype"] = KV_DTYPE
    if WEIGHT_DTYPE:
        additional["experimental_weight_dtype"] = WEIGHT_DTYPE
    exp = os.environ.get("BD_EXPORT")
    if exp:
        additional["export_path"] = exp
        additional["export_model_name"] = os.environ.get("BD_EXPORT_NAME", "bd")

    llm = vllm.LLM(
        model="meta-llama/Llama-3.2-3B-Instruct",
        max_model_len=MML,
        max_num_seqs=BATCH,
        # budget = batch*MML so prompts never chunk (legacy single-shot prefill).
        max_num_batched_tokens=BATCH * MML,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
        additional_config=additional,
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
    outs = llm.generate([PROMPT] * BATCH, sp)
    seqs = [tuple(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    print(
        f"BD_BATCH={BATCH} BD_MML={MML} BD_CPU_SAMPLING={int(CPU_SAMPLING)} "
        f"BD_NLAYERS={NLAYERS} prompt_tokens={len(outs[0].prompt_token_ids)}"
    )
    for i, (s, t) in enumerate(zip(seqs, texts)):
        print(f"ROW[{i}] ids={list(s)}")
        print(f"ROW[{i}] text={t!r}")
    n_unique = len(set(seqs))
    print(f"RESULT unique_sequences={n_unique} all_identical={n_unique == 1}")


if __name__ == "__main__":
    main()
