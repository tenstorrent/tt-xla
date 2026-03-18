# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import vllm


@pytest.mark.nightly
@pytest.mark.single_device
def test_tinyllama_generation():
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8)
    llm_args = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.05,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }

    print(f"\n=== Loading model: {llm_args['model']} ===")
    t0 = time.perf_counter()
    llm = vllm.LLM(**llm_args)
    print(f"=== Model loaded in {time.perf_counter() - t0:.2f}s ===")

    print(f"=== Starting generation (max_tokens={sampling_params.max_tokens}) ===")
    t0 = time.perf_counter()
    output = llm.generate(prompts, sampling_params)[0]
    elapsed = time.perf_counter() - t0
    output_text = output.outputs[0].text
    num_tokens = len(output.outputs[0].token_ids)
    print(
        f"=== Generation done: {num_tokens} tokens in {elapsed:.2f}s ({num_tokens / elapsed:.2f} tok/s) ==="
    )
    print(f"prompt: {prompts[0]}, output: {output_text}")
