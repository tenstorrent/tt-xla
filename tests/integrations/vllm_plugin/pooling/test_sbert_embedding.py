# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import vllm


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    ["model_name", "baseline_path"],
    [
        pytest.param(
            "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            "baseline/bert_base_turkish_cased_mean_nli_stsb_tr_baseline.pt",
        ),
        pytest.param(
            "sentence-transformers/all-MiniLM-L6-v2",
            "baseline/all_MiniLM_L6_v2_baseline.pt",
        ),
        pytest.param(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "baseline/multi_qa_MiniLM_L6_cos_v1_baseline.pt",
        ),
    ],
)
def test_embed_bge(model_name: str, baseline_path):
    """
    Test the sentence-Bert models' embedding outputs for correctness.
    Baseline embeddings are computed using vLLM on CPU backend.
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
        "max_model_len": 75,
        "disable_sliding_window": True,
        "max_num_batched_tokens": 75,
        "max_num_seqs": 2,
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
