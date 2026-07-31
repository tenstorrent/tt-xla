"""Minimal-warmup DP probe -- same bug, far less compile time.

The full probe spends most of an hour warming graph shapes we do not need. The
DP continuation-chunk corruption needs only: dp >= 2, >= 2 rows, and a prompt
long enough to split into a second chunk. Everything else is incidental, so this
strips it:

  - facebook/opt-125m instead of Qwen3-0.6B  (~5x fewer layers to trace)
  - max_model_len=256, chunk=32              (smallest legal value: chunked
                                              prefill needs max_num_blocks_per_req
                                              % 8 == 0, i.e. max_model_len a
                                              multiple of 8*block_size = 256)
  - enable_precompile_all=False              (skip warming every shape)
  - optimization_level=0                     (skip optimizer passes)
  - max_tokens=1                             (prefill is where it breaks;
                                              first_diff=0 established that)

Run with TTXLA_DP_DEBUG=1 to get the per-step instrumentation.

usage: dp_probe_fast.py [num_reqs] [chunk] [reps]
  defaults: 2 32 4   -> 2 identical prompts, ~55 tokens => 2 chunks at chunk=32

IMPORTANT: this config is only useful if it still reproduces. Row 1 must come
out degenerate. If all rows look correct here but the full probe corrupts, the
shrink removed whatever the bug needs -- add dimensions back (bigger model, more
heads, larger chunk) rather than concluding anything is fixed.
"""
import os
import re
import sys

import vllm

BASE = (
    "The history of computing spans many centuries and each generation "
    "improved speed. "
)
WORD_RE = re.compile(r"[A-Za-z']+")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    prompts = [BASE * reps] * n

    llm = vllm.LLM(
        model="facebook/opt-125m",
        max_num_seqs=n,
        max_model_len=256,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=True,
        additional_config={
            "min_context_len": 32,
            "enable_data_parallel": True,
            "prefill_chunk_size": chunk,
            "cpu_sampling": True,
            # Warmup reduction -- the point of this script.
            "enable_precompile_all": False,
            "optimization_level": 0,
        },
    )
    # 4 tokens, not 1: enough to see a degenerate run-on, still cheap.
    out = llm.generate(prompts, vllm.SamplingParams(temperature=0.0, max_tokens=4))
    ids = [list(o.outputs[0].token_ids) for o in out]

    print("", flush=True)
    ref = ids[0]
    print(f"row 0 (reference): ids={ref}  {out[0].outputs[0].text[:48]!r}", flush=True)
    mismatched = []
    for i in range(1, n):
        first_diff = next(
            (k for k in range(min(len(ref), len(ids[i]))) if ref[k] != ids[i][k]),
            None,
        )
        if ids[i] != ref:
            mismatched.append(i)
        print(
            f"row {i}: first_diff={first_diff} ids={ids[i]}  "
            f"{out[i].outputs[0].text[:48]!r}",
            flush=True,
        )
    print(
        f"\nRESULT n={n} chunk={chunk} reps={reps}: mismatched rows={mismatched}\n"
        f"(identical prompts + greedy => every row MUST match row 0; "
        f"a non-empty list reproduces the bug)",
        flush=True,
    )
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
