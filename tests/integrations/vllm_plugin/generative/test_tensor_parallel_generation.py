# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.dual_chip
@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-3B"])
def test_tensor_parallel_generation_n300(model_name: str):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.skip(
    reason="Skipping due to bug in tt-metal SDPA op. Issue: https://github.com/tenstorrent/tt-xla/issues/3465"
)
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_enable_weight_bfp8_conversion"],
    [
        pytest.param("Qwen/Qwen3-0.6B", False, False),
    ],
)
def test_tensor_parallel_generation_llmbox_small(
    model_name: str,
    enable_const_eval: bool,
    experimental_enable_weight_bfp8_conversion: bool,
):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_enable_weight_bfp8_conversion": experimental_enable_weight_bfp8_conversion,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
# @pytest.mark.skip(
#     reason="Skipping due to bug in tt-metal SDPA op. Issue: https://github.com/tenstorrent/tt-xla/issues/3465"
# )
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_enable_weight_bfp8_conversion"],
    [
        # pytest.param("Qwen/Qwen3-32B", False, False),
        # pytest.param("Qwen/Qwen2.5-32B", False, False),
        # pytest.param("meta-llama/Llama-3.1-70B", True, True),
        pytest.param("mistralai/Mistral-Small-3.2-24B-Instruct-2506", False, False),
    ],
)
def test_tensor_parallel_generation_llmbox_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_enable_weight_bfp8_conversion: bool,
):
    image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

    # Same message format as Transformers / OpenAI chat (vLLM accepts this for .chat())
    user_text = (
        "What action do you think I should take in this situation? "
        "List all the possible actions and explain why you think they are good or bad."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)

    llm_args = {
        "model": model_name,
        "tokenizer_mode": "mistral",
        "limit_mm_per_prompt": {"image": 4},
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 2048,
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_enable_weight_bfp8_conversion": experimental_enable_weight_bfp8_conversion,
        },
    }
    llm = vllm.LLM(**llm_args)

    completion = llm.chat(messages, sampling_params=sampling_params)
    # vLLM chat returns OpenAI-style completion: choices[0].message.content
    if hasattr(completion, "choices") and completion.choices:
        output_text = completion.choices[0].message.content
    elif (
        isinstance(completion, list)
        and completion
        and hasattr(completion[0], "content")
    ):
        output_text = completion[0].content
    else:
        output_text = str(completion)
    print(f"prompt: {user_text[:80]}..., output: {output_text}")
