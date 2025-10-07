# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import vllm

import pytest


@pytest.mark.push
def test_embed_qwen3():
    prompts = [
        "Hello, my name is",
    ]
    llm_args = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "task": "embed",
        "dtype": "bfloat16",
        "enforce_eager": True,
        "max_model_len": 64,
        "disable_sliding_window": True,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 1,
    }
    model = vllm.LLM(**llm_args)

    output_embedding = model.embed(prompts)
    print(f"prompt: {prompts[0]}, output: {output_embedding}")
