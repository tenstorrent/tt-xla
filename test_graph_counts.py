# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Graph count validation for PR #4789 (reduce vLLM decode graphs 5→2).

Runs 4 combinations:
  - opt-125m,    cpu_sampling=False  (fused decode_postprocess path)
  - opt-125m,    cpu_sampling=True   (cpu sampling path)
  - Qwen3-0.6B (TP), cpu_sampling=False
  - Qwen3-0.6B (TP), cpu_sampling=True

Set VLLM_XLA_CHECK_RECOMPILATION=1 before running to see graph count logs:
  VLLM_XLA_CHECK_RECOMPILATION=1 python3 test_graph_counts.py
"""
import gc
import os
import sys

import vllm

PROMPT = "Hello, my name is"
MAX_TOKENS = 16

# opt-125m single-chip cases must run on n150 — this is an n300 (2-chip) machine
# and the TT binary cache gets populated with 2-device binaries causing device count
# mismatch errors. Run those cases via CI (run_vllm_n150_tests job).
CASES = [
    {
        "label": "Qwen/Qwen3-0.6B (TP) | cpu_sampling=False",
        "model": "Qwen/Qwen3-0.6B",
        "cpu_sampling": False,
        "tp": True,
    },
    {
        "label": "Qwen/Qwen3-0.6B (TP) | cpu_sampling=True",
        "model": "Qwen/Qwen3-0.6B",
        "cpu_sampling": True,
        "tp": True,
    },
]


def run_case(case: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {case['label']}")
    print(f"{'='*70}\n")

    # Single-chip cases: restrict to device 0 only so the TT runtime doesn't
    # compile for 2 devices (n300 has 2 chips; this avoids device count mismatch).
    if not case["tp"]:
        os.environ["TT_VISIBLE_DEVICES"] = "0"
    else:
        os.environ.pop("TT_VISIBLE_DEVICES", None)

    additional_config = {
        "enable_const_eval": False,
        "min_context_len": 32,
        "num_hidden_layers": 1,
        "cpu_sampling": case["cpu_sampling"],
    }
    if case["tp"]:
        additional_config["enable_tensor_parallel"] = True
        additional_config["use_2d_mesh"] = True

    llm = vllm.LLM(
        model=case["model"],
        max_num_batched_tokens=64,
        max_num_seqs=1,
        max_model_len=64,
        gpu_memory_utilization=0.002,
        additional_config=additional_config,
    )

    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    output = llm.generate([PROMPT], sampling_params)[0].outputs[0].text
    print(f"\n>>> Output: {output!r}\n")

    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()


if __name__ == "__main__":
    if not os.environ.get("VLLM_XLA_CHECK_RECOMPILATION"):
        print(
            "WARNING: VLLM_XLA_CHECK_RECOMPILATION not set — graph counts won't be logged."
        )
        print("Run as: VLLM_XLA_CHECK_RECOMPILATION=1 python3 test_graph_counts.py\n")

    failures = []
    for case in CASES:
        try:
            run_case(case)
        except Exception as e:
            print(f"\nFAILED [{case['label']}]: {e}\n", file=sys.stderr)
            failures.append((case["label"], e))

    print(f"\n{'='*70}")
    if failures:
        print(f"FAILED {len(failures)}/{len(CASES)} cases:")
        for label, err in failures:
            print(f"  - {label}: {err}")
        sys.exit(1)
    else:
        print(f"ALL {len(CASES)} cases passed.")
