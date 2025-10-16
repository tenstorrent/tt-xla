# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import vllm


@pytest.mark.push
def test_embed_qwen3():
    # max_seq_len = 2**14
    max_seq_len = 2**7
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
        # "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
    }
    model = vllm.LLM(**llm_args)

    for seq_len, prompts in prompts_list:
        output_embedding = model.embed(prompts)
        start_time = time.time()
        output_embedding = model.embed(prompts)
        end_time = time.time()
        print(f"seq_len: {seq_len}, time: {end_time - start_time}")
        print(f"prompt: {prompts[0]}, output: {output_embedding}")

    # output_embedding = model.embed(prompts)
    # print(f"prompt: {prompts[0]}, output: {output_embedding}")
