# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import time

import pytest
import torch
import vllm


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    ["model_name", "baseline_path"],
    [
        pytest.param(
            "Qwen/Qwen3-Embedding-4B",
            "baseline/qwen3_embedding_4B_baseline.pt",
        ),
        pytest.param(
            "Qwen/Qwen3-Embedding-0.6B",
            "baseline/qwen3_embedding_0.6B_baseline.pt",
        ),
    ],
)
def test_embed_qwen3(model_name: str, baseline_path: str):
    """
    Test the Qwen3-Embedding models embedding outputs for correctness
    under different batching and padding scenarios.

    Test Setup:
    - Input consists of four prompts with token lengths [32, 15, 29, 7].
    - vLLM is configured with max_num_seqs=2, meaning each batch can contain
      up to 2 sequences and vLLM always process single prompt in first batch.
    - This results in three batches:
        1. First batch: first prompt (32 tokens) → no padding required.
        2. Second batch: second and third prompts concatenated (15 + 29 = 44
           tokens), padded to max_model_len=64.
        3. Third batch: fourth prompt (7 tokens), padded to max_model_len=32.

    Purpose:
    - Validates that the model produces embeddings consistent with precomputed
      baseline embeddings for each prompt.
    - Ensures Pearson Correlation Coefficient (PCC) > 0.99 for each embedding.

    Baseline Embeddings:
    - Baseline embeddings are computed using vLLM on CPU backend and stored in
      'baseline' directory.
    """

    path = os.path.join(os.path.dirname(__file__), baseline_path)
    loaded_data = torch.load(path)

    prompts = [
        "The quick-thinking engineer designed a compact neural processor that could adapt to changing data patterns in real time, optimizing energy use while maintaining exceptional computational accuracy as well.",
        "Hello, my name is chatbot. How can I help you?",
        "We build computers for AI. We design Graph Processors, high-performance RISC CPUs, and configurable chips that run our robust software stack.",
        "The capital of France is Paris",
    ]
    llm_args = {
        "model": model_name,
        "task": "embed",
        "dtype": "bfloat16",
        "max_model_len": 64,
        "disable_sliding_window": True,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 2,
    }
    model = vllm.LLM(**llm_args)

    output_embedding = model.embed(prompts)

    for idx, (prompt, output) in enumerate(zip(prompts, output_embedding)):
        embeds = output.outputs.embedding
        embeds_trimmed = (
            (str(embeds[:32])[:-1] + ", ...]") if len(embeds) > 32 else embeds
        )
        print(f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})")

        output_tensor = torch.tensor(embeds, dtype=torch.float32)
        golden_tensor = loaded_data[f"prompt{idx}"]
        pcc = torch.corrcoef(torch.stack([output_tensor, golden_tensor]))[0, 1]
        print("PCC:", pcc.item())
        assert pcc.item() > 0.99, f"PCC Error: Incorrect embedding for prompt{idx}"

        print("-" * 60)


@pytest.mark.nightly
@pytest.mark.single_device
def test_embed_qwen3_perf():
    max_seq_len = 2**14  # 16384
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

    # Precompile of model backbone done here
    model = vllm.LLM(**llm_args)

    # Precompile pre/post processing graphs which are part of the actual user flow
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


@pytest.mark.push
@pytest.mark.single_device
def test_embed_qwen3_reduced_dims():
    prompts = [
        "Hello, my name is",
    ]
    llm_args = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "task": "embed",
        "dtype": "bfloat16",
        "max_model_len": 64,
        "disable_sliding_window": True,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 1,
        "hf_overrides": {"is_matryoshka": True},
    }
    model = vllm.LLM(**llm_args)

    pooling_params = vllm.PoolingParams(dimensions=128)

    output_embedding = model.embed(prompts, pooling_params=pooling_params)
    print(f"Prompt: {prompts[0]}")
    print(f"Embeddings: {output_embedding[0].outputs.embedding}")
    print(f"len={len(output_embedding[0].outputs.embedding)}")
    assert (
        len(output_embedding[0].outputs.embedding) == 128
    ), f"vLLM generated incorrect number of embedding dimensions; Expected dims=128 but got {len(output_embedding[0].outputs.embedding)}"


@pytest.mark.push
@pytest.mark.single_device
def test_embed_qwen3_8K():
    # Enable program cache for better performance
    seq_len = 2**13  # 8192
    prompt = ["hello " * (seq_len - 2)]

    llm_args = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "task": "embed",
        "dtype": "bfloat16",
        "max_model_len": seq_len,
        "disable_sliding_window": True,
        "max_num_batched_tokens": seq_len,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": seq_len,
        },
    }

    # Precompile of model backbone done here
    model = vllm.LLM(**llm_args)

    output_embedding = model.embed(prompt)
    print(f"Finished precompile for seq_len: {seq_len}")


@pytest.mark.push
@pytest.mark.single_device
def test_embed_qwen3_16K():
    # Enable program cache for better performance
    seq_len = 2**14  # 16384
    prompt = ["hello " * (seq_len - 2)]

    llm_args = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "task": "embed",
        "dtype": "bfloat16",
        "max_model_len": seq_len,
        "disable_sliding_window": True,
        "max_num_batched_tokens": seq_len,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": seq_len,
        },
    }

    # Precompile of model backbone done here
    model = vllm.LLM(**llm_args)

    output_embedding = model.embed(prompt)
    print(f"Finished precompile for seq_len: {seq_len}")


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.parametrize("batch_size", [2, 4])
def test_embed_qwen3_data_parallel(batch_size: int):
    """
    Test the Qwen3-Embedding-4B model for data parallel.
    """

    baseline_path = "baseline/qwen3_embedding_4B_baseline.pt"
    path = os.path.join(os.path.dirname(__file__), baseline_path)
    print(f"Loading baseline embeddings from: {path} \n {os.path.exists(path)} ")

    loaded_data = torch.load(path)

    prompts = [
        "The quick-thinking engineer designed a compact neural processor that could adapt to changing data patterns in real time, optimizing energy use while maintaining exceptional computational accuracy as well.",
        "Hello, my name is chatbot. How can I help you?",
        "We build computers for AI. We design Graph Processors, high-performance RISC CPUs, and configurable chips that run our robust software stack.",
        "The capital of France is Paris",
    ]
    llm_args = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "task": "embed",
        "dtype": "bfloat16",
        "max_model_len": 64,
        "disable_sliding_window": True,
        "max_num_batched_tokens": 64,
        "max_num_seqs": batch_size,
        "additional_config": {
            "batch_size": batch_size,
            "is_data_parallel": True,
        },
    }
    model = vllm.LLM(**llm_args)

    output_embedding = model.embed(prompts)

    pcc_values = []
    for idx, (prompt, output) in enumerate(zip(prompts, output_embedding)):
        embeds = output.outputs.embedding
        embeds_trimmed = (
            (str(embeds[:32])[:-1] + ", ...]") if len(embeds) > 32 else embeds
        )
        print(f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})")

        output_tensor = torch.tensor(embeds, dtype=torch.float32)
        golden_tensor = loaded_data[f"prompt{idx}"]
        pcc = torch.corrcoef(torch.stack([output_tensor, golden_tensor]))[0, 1]
        print("PCC:", pcc.item())
        pcc_values.append(pcc.item())
        print("-" * 60)

    assert all(p >= 0.99 for p in pcc_values), f"Low PCC values: {pcc_values}"
