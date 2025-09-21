# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import vllm

import pytest


@pytest.mark.push
def test_opt_generation():
    prompts = [
        "Hello, my name is",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
    llm_args = {
        "model": "Qwen/Qwen3-4B-Thinking-2507",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    # output = llm.embed(prompts)
    print(f"prompt: {prompts[0]}, output: {output_text}")
