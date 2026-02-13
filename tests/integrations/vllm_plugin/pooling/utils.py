# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import vllm


def run_pooling_test(
    model_name: str,
    baseline_path,
    max_model_len: int,
    experimental_enable_weight_bfp8_conversion: bool = False,
    enable_tensor_parallel: bool = False,
    enable_data_parallel: bool = False,
    min_context_len: int = 128,
    enable_const_eval: bool = True,
    batch_size: int = 1,
    max_num_reqs: int = 2,
    max_num_batched_tokens: int = 128,
    optimization_level: int = 0,
):
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
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "disable_sliding_window": True,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_reqs,
        "additional_config": {
            "experimental_enable_weight_bfp8_conversion": experimental_enable_weight_bfp8_conversion,
            "enable_tensor_parallel": enable_tensor_parallel,
            "enable_data_parallel": enable_data_parallel,
            "min_context_len": min_context_len,
            "enable_const_eval": enable_const_eval,
            "batch_size": batch_size,
            "optimization_level": optimization_level,
        },
    }
    model = vllm.LLM(**llm_args)

    output_embedding = model.embed(prompts)

    pcc_values = []
    for idx, (prompt, output) in enumerate(zip(prompts, output_embedding)):
        embeds = output.outputs.embedding
        embeds_trimmed = (
            (str(embeds[:32])[:-1] + ", ...]") if len(embeds) > 16 else embeds
        )
        print(f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})")

        output_tensor = torch.tensor(embeds, dtype=torch.float32)
        golden_tensor = loaded_data[f"prompt{idx}"]
        pcc = torch.corrcoef(torch.stack([output_tensor, golden_tensor]))[0, 1]
        print("PCC:", pcc.item())
        print("-" * 60)
        pcc_values.append(pcc.item())

    assert all(p >= 0.99 for p in pcc_values), f"Low PCC values: {pcc_values}"
