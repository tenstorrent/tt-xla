# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import vllm


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.parametrize(
    ["model_name", "baseline_path"],
    [
        pytest.param(
            "BAAI/bge-m3",
            "baseline/bge_m3_baseline.pt",
        ),
        pytest.param(
            "Qwen/Qwen3-Embedding-4B",
            "baseline/qwen3_embedding_4B_baseline.pt",
        ),
    ],
)
@pytest.mark.parametrize(
    "batch_size, max_num_seqs, max_num_batched_tokens",
    [
        (2, 2, 64),
        (4, 4, 128),
    ],
)
def test_data_parallel_inference(
    model_name: str,
    baseline_path: str,
    batch_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
):
    """
    Test data parallel inference with vLLM for embedding models.
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
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
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
            (str(embeds[:16])[:-1] + ", ...]") if len(embeds) > 16 else embeds
        )
        print(f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})")

        output_tensor = torch.tensor(embeds, dtype=torch.float32)
        golden_tensor = loaded_data[f"prompt{idx}"]
        pcc = torch.corrcoef(torch.stack([output_tensor, golden_tensor]))[0, 1]
        print("PCC:", pcc.item())
        pcc_values.append(pcc.item())
        print("-" * 60)

    assert all(p >= 0.99 for p in pcc_values), f"Low PCC values: {pcc_values}"
