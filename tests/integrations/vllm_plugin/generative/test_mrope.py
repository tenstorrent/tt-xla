# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory


@pytest.mark.push
@pytest.mark.single_device
def test_mrope():
    prompts = [
        "Continue in English: I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model_name = "Qwen/Qwen3.6-27B"

    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.2,
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        "additional_config": {
            "min_context_len": 32,
            # "num_hidden_layers": 4,
            "enable_tensor_parallel": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    # assert_output_coherent(output_text)

    check_host_memory(model_name)
