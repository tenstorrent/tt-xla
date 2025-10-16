# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import vllm


@pytest.mark.push
def test_embed_qwen3():
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    llm_args = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "task": "embed",
        "dtype": "bfloat16",
        "max_model_len": 64,
        "disable_sliding_window": True,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 1,
    }
    model = vllm.LLM(**llm_args)

    output_embedding = model.embed(prompts)
    print(f"prompt: {prompts[0]}, output: {output_embedding}")


@pytest.mark.push
def test_embed_qwen3_perf():
    max_seq_len = 2**14
    prompts_list = []

    i = 128
    while i <= max_seq_len:
        prompts_list.append((i, ["hello " * (i - 2)]))
        i *= 2
    llm_args = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "task": "embed",
        "dtype": "bfloat16",
        "max_model_len": max_seq_len,
        "disable_sliding_window": True,
        "max_num_batched_tokens": max_seq_len,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "additional_config": {
            "enable_const_eval": False,
        },
    }
    model = vllm.LLM(**llm_args)

    # Precompile executions
    for seq_len, prompts in prompts_list:
        output_embedding = model.embed(prompts)
        print(f"Finished precompile for seq_len: {seq_len}")

    perf_data = {}
    # Benchmark E2E latency
    for seq_len, prompts in prompts_list:
        start_time = time.time()
        output_embedding = model.embed(prompts)
        end_time = time.time()
        perf_data[seq_len] = end_time - start_time
        print(f"seq_len: {seq_len}, time: {end_time - start_time}")

    print("Latency per sequence length:")
    for seq_len, latency in perf_data.items():
        print(f"seq_len: {seq_len}, latency: {latency}")
