#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Capture real model logits for use in test_sampling_with_saved_logits.py.

Runs Llama-3.2-3B on a fixed prompt and saves the logits at the first decode
step. Uses TT_CAPTURE_LOGITS_PATH env var to trigger saving inside the engine
subprocess (monkey-patching the main process doesn't work because vLLM v1
runs the model in a separate EngineCore process).

Usage:
    python tests/integrations/vllm_plugin/sampling/capture_logits.py

Output:
    tests/integrations/vllm_plugin/sampling/fixtures/llama3_2_3b_decode_step1.pt
"""

import os
import sys

import torch
import vllm

PROMPT = "Tell me a short story about a fox and a river."
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "llama3_2_3b_decode_step1.pt"
)


def main():
    print(f"Model : {MODEL}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Output: {OUTPUT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.environ["TT_CAPTURE_LOGITS_PATH"] = OUTPUT_PATH

    llm = vllm.LLM(
        model=MODEL,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.05,
        additional_config={
            "enable_const_eval": True,
            "min_context_len": 32,
        },
    )

    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=10)
    llm.generate([PROMPT], sampling_params)

    if not os.path.exists(OUTPUT_PATH):
        print("ERROR: logits were not captured. Did the model run?")
        sys.exit(1)

    fixture = torch.load(OUTPUT_PATH, weights_only=False)
    fixture["prompt"] = PROMPT
    fixture["model"] = MODEL
    torch.save(fixture, OUTPUT_PATH)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nSaved {OUTPUT_PATH} ({size_kb:.0f} KB)")
    print(f"  vocab_size   = {fixture['logits'].shape[-1]}")
    print(f"  greedy_token = {fixture['greedy_token']}")


if __name__ == "__main__":
    main()
