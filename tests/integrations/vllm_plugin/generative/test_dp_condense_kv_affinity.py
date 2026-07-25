# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm

PROMPTS = [
    "Continue in English: The capital of France is",
    "Continue in English: Water is made of hydrogen and",
    "Continue in English: The opposite of hot is",
    "Continue in English: Photosynthesis is the process by which plants",
    "Continue in English: The largest planet in our solar system is",
    "Continue in English: A group of wolves is called a",
    "Continue in English: The first president of the United States was",
    "Continue in English: The chemical symbol for gold is",
]


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-8B"])
def test_dp_condense_kv_affinity(model_name: str):
    n = len(PROMPTS)
    long_len, short_len = 40, 4
    survivors = list(range(n // 2, n))

    llm = vllm.LLM(
        model=model_name,
        max_num_batched_tokens=8192,
        max_num_seqs=n,
        max_model_len=128,
        gpu_memory_utilization=0.3,
        additional_config={
            "enable_const_eval": True,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "enable_data_parallel": True,
            "shard_weights_on_batch_axis": True,
            "experimental_weight_dtype": "",
            "mesh_shape": [2, 4],
            "cpu_sampling": False,
        },
    )

    ref = llm.generate(
        PROMPTS,
        [
            vllm.SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=long_len, ignore_eos=True
            )
            for _ in range(n)
        ],
    )
    test = llm.generate(
        PROMPTS,
        [
            vllm.SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=(short_len if i < n // 2 else long_len),
                ignore_eos=True,
            )
            for i in range(n)
        ],
    )

    for i in survivors:
        ref_ids = list(ref[i].outputs[0].token_ids)
        test_ids = list(test[i].outputs[0].token_ids)
        m = min(len(ref_ids), len(test_ids))
        assert (
            test_ids[:m] == ref_ids[:m]
        ), f"survivor position {i} diverges from the clean reference (#5778)"
