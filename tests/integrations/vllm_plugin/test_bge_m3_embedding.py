# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import time

import pytest
import torch
import vllm


@pytest.mark.push
def test_embed_bge_m3():
    """
    Test the BGE-M3 model's embedding outputs for correctness
    under different batching and padding scenarios.

    Test Setup:
    - Input consists of four prompts with varying token lengths.
    - vLLM is configured with max_num_seqs=2, meaning each batch can contain
      up to 2 sequences and vLLM always process single prompt in first batch.
    - This results in three batches:
        1. First batch: first prompt → no padding required.
        2. Second batch: second and third prompts concatenated,
           padded to max_model_len=512.
        3. Third batch: fourth prompt, padded to appropriate length.

    Purpose:
    - Validates that the model produces embeddings consistent with precomputed
      baseline embeddings for each prompt.
    - Ensures Pearson Correlation Coefficient (PCC) > 0.99 for each embedding.

    Baseline Embeddings:
    - Baseline embeddings are computed using vLLM on CPU backend and stored as
      'bge_m3_embedding_baseline.pt' file.
    """

    prompts = [
        "The quick-thinking engineer designed a compact neural processor that could adapt to changing data patterns in real time, optimizing energy use while maintaining exceptional computational accuracy as well.",
        "Hello, my name is chatbot. How can I help you?",
        "We build computers for AI. We design Graph Processors, high-performance RISC CPUs, and configurable chips that run our robust software stack.",
        "The capital of France is Paris",
    ]
    llm_args = {
        "model": "BAAI/bge-m3",
        "task": "embed",
        "dtype": "bfloat16",
        "max_model_len": 512,
        "disable_sliding_window": True,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 2,
    }
    model = vllm.LLM(**llm_args)

    output_embedding = model.embed(prompts)

    path = os.path.join(os.path.dirname(__file__), "bge_m3_embedding_baseline.pt")
    loaded_data = torch.load(path)

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


@pytest.mark.push
def test_embed_bge_m3_perf():
    """
    Performance test for BGE-M3 model's embedding generation.

    This test measures the end-to-end latency for generating embeddings
    at different sequence lengths, from 128 tokens up to 8192 tokens.

    Test Setup:
    - Uses BGE-M3 model with max_model_len=8192 (based on service.sh config)
    - Tests sequence lengths: 128, 256, 512, 1024, 2048, 4096, 8192
    - Performs precompilation first, then measures actual inference latency
    """
    max_seq_len = 2**13  # 8192 (BGE-M3 max context length)
    prompts_list = []

    i = 128
    while i <= max_seq_len:
        num_hellos = max(1, (i // 2 - 2))  # Hello is ~2 tokens for bge-m3
        prompts_list.append(
            (i, [f"hello " * num_hellos + "hello"])
        )  # hello can't make exact token count
        i *= 2

    llm_args = {
        "model": "BAAI/bge-m3",
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

    # Test actual token counts
    print("Verifying token counts:")
    for seq_len, prompts in prompts_list:
        prompt = prompts[0]  # Get the actual prompt string
        # Use vLLM's tokenizer to get actual token count
        token_ids = model.llm_engine.tokenizer.encode(prompt)
        actual_tokens = len(token_ids)
        print(f"Target: {seq_len}, Actual: {actual_tokens}, Prompt: '{prompt[:50]}...'")

        # Optional: Assert that we're within reasonable bounds
        assert (
            actual_tokens <= max_seq_len
        ), f"Prompt too long: {actual_tokens} > {max_seq_len}"

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
