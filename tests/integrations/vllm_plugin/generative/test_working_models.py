# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test vLLM generative models on single-chip Blackhole.

Covers a range of model families and sizes to validate the SDPA decode path.
Models with >16 query heads previously hit a BH-specific SFPI register spill
bug (tt-xla #3803). The tt-mlir fix (PR #7561) disables exp_approx_mode for
BH paged SDPA decode, which should unblock all models.

Run all:
    pytest -sv tests/integrations/vllm_plugin/generative/test_working_models.py

Run one:
    pytest -sv "tests/integrations/vllm_plugin/generative/test_working_models.py::test_vllm_generation[llama3.2-3b]"
"""

import time

import pytest
import vllm


# Model registry: (model_name, query_heads, notes)
# <=16 query heads: previously unaffected by #3803
# >16 query heads: previously blocked by #3803, now fixed by tt-mlir #7561
MODELS = [
    # --- Small models (fast iteration) ---
    pytest.param(
        "facebook/opt-125m", 12, 0.05,
        id="opt-125m",
        marks=[pytest.mark.push],
    ),
    pytest.param(
        "Qwen/Qwen2.5-0.5B-Instruct", 16, 0.05,
        id="qwen2.5-0.5b",
    ),
    pytest.param(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 32, 0.05,
        id="tinyllama-1.1b",
    ),
    # --- Medium models ---
    pytest.param(
        "Qwen/Qwen2.5-1.5B-Instruct", 12, 0.05,
        id="qwen2.5-1.5b",
    ),
    pytest.param(
        "meta-llama/Llama-3.2-1B-Instruct", 32, 0.05,
        id="llama3.2-1b",
    ),
    pytest.param(
        "meta-llama/Llama-3.2-3B-Instruct", 24, 0.05,
        id="llama3.2-3b",
    ),
    pytest.param(
        "Qwen/Qwen2.5-3B-Instruct", 16, 0.05,
        id="qwen2.5-3b",
    ),
    # --- Large models (demo targets) ---
    pytest.param(
        "Qwen/Qwen2.5-7B-Instruct", 28, 0.05,
        id="qwen2.5-7b",
        marks=[pytest.mark.nightly],
    ),
    pytest.param(
        "meta-llama/Llama-3.1-8B-Instruct", 32, 0.05,
        id="llama3.1-8b",
        marks=[pytest.mark.nightly],
    ),
]


@pytest.mark.single_device
@pytest.mark.parametrize("model_name,query_heads,gpu_mem_util", MODELS)
def test_vllm_generation(model_name, query_heads, gpu_mem_util):
    prompts = [
        "Explain quantum computing in one sentence.",
    ]
    sampling_params = vllm.SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=32
    )
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": gpu_mem_util,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }

    print(
        f"\n=== Loading model: {model_name} "
        f"(query_heads={query_heads}) ==="
    )
    t0 = time.perf_counter()
    llm = vllm.LLM(**llm_args)
    load_time = time.perf_counter() - t0
    print(f"=== Model loaded in {load_time:.2f}s ===")

    print(f"=== Starting generation (max_tokens={sampling_params.max_tokens}) ===")
    t0 = time.perf_counter()
    output = llm.generate(prompts, sampling_params)[0]
    elapsed = time.perf_counter() - t0
    output_text = output.outputs[0].text
    num_tokens = len(output.outputs[0].token_ids)
    print(
        f"=== Generation done: {num_tokens} tokens in {elapsed:.2f}s "
        f"({num_tokens / elapsed:.2f} tok/s) ==="
    )
    print(f"prompt: {prompts[0]}")
    print(f"output: {output_text}")
