# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import time

import pytest
import torch
import vllm

from tests.integrations.vllm_plugin.pooling.utils import run_pooling_test


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    ["model_name", "baseline_path", "max_model_len"],
    [
        pytest.param(
            "Qwen/Qwen3-Embedding-0.6B",
            "baseline/qwen3_embedding_0.6B_baseline.pt",
            75,
        ),
    ],
)
def test_embedding_push(model_name: str, baseline_path, max_model_len: int):
    run_pooling_test(model_name, baseline_path, max_model_len, min_context_len=32)


@pytest.mark.push
@pytest.mark.single_device
def test_embed_qwen3_reduced_dims():
    prompts = [
        "Hello, my name is",
    ]
    llm_args = {
        "model": "Qwen/Qwen3-Embedding-0.6B",
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


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    [
        "model_name",
        "baseline_path",
        "max_model_len",
        "experimental_weight_dtype",
    ],
    [
        pytest.param(
            "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            "baseline/bert_base_turkish_cased_mean_nli_stsb_tr_baseline.pt",
            75,
            False,
        ),
        pytest.param(
            "sentence-transformers/all-MiniLM-L6-v2",
            "baseline/all_MiniLM_L6_v2_baseline.pt",
            75,
            False,
        ),
        pytest.param(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "baseline/multi_qa_MiniLM_L6_cos_v1_baseline.pt",
            75,
            False,
        ),
        pytest.param(
            "BAAI/bge-m3",
            "baseline/bge_m3_baseline.pt",
            64,
            False,
        ),
        pytest.param(
            "BAAI/bge-base-en",
            "baseline/bge_base_en_baseline.pt",
            64,
            False,
        ),
        pytest.param(
            "BAAI/bge-large-en",
            "baseline/bge_large_en_baseline.pt",
            64,
            False,
        ),
        pytest.param(
            "BAAI/bge-small-en",
            "baseline/bge_small_en_baseline.pt",
            64,
            False,
        ),
        pytest.param(
            "intfloat/e5-base-v2",
            "baseline/e5_base_v2_baseline.pt",
            64,
            False,
        ),
        pytest.param(
            "intfloat/e5-large-v2",
            "baseline/e5_large_v2_baseline.pt",
            64,
            False,
        ),
        pytest.param(
            "intfloat/e5-small-v2",
            "baseline/e5_small_v2_baseline.pt",
            64,
            False,
        ),
        pytest.param(
            "Qwen/Qwen3-Embedding-4B",
            "baseline/qwen3_embedding_4B_baseline.pt",
            64,
            False,
        ),
        pytest.param(
            "Qwen/Qwen3-Embedding-8B",
            "baseline/qwen3_embedding_8B_baseline.pt",
            64,
            True,
            marks=pytest.mark.xfail(
                reason="Static CBs exceed L1 size - https://github.com/tenstorrent/tt-xla/issues/2935"
            ),
        ),
    ],
)
def test_embedding_nightly(
    model_name: str,
    baseline_path,
    max_model_len: int,
    experimental_weight_dtype: str,
):
    run_pooling_test(
        model_name,
        baseline_path,
        max_model_len,
        experimental_weight_dtype,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    ["model_name", "baseline_path", "optimization_level"],
    [
        pytest.param(
            "BAAI/bge-m3",
            "baseline/bge_m3_baseline.pt",
            0,
        ),
        pytest.param(
            "Qwen/Qwen3-Embedding-0.6B",
            "baseline/qwen3_embedding_0.6B_baseline.pt",
            1,
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
def test_batched_inference(
    model_name: str,
    baseline_path: str,
    optimization_level: int,
    batch_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
):
    """
    Test multi-batched inputs. Runner will create inputs of shape [batch_size x input_len]
    - BGE-m3: Model with encoder-only attention layers.
    - Qwen3-Embedding-0.6B: Model with decoder-only attention layers.
    Note:
      - max_model_len * max_num_seqs <= max_num_batched_tokens
      - max_num_reqs == batch_size

    - Baseline embeddings are computed using vLLM on CPU backend.
    """
    if model_name == "Qwen/Qwen3-Embedding-0.6B" and batch_size == 2:
        pytest.skip(
            "Skipping due to non-deterministic failure in CI. Issue: https://github.com/tenstorrent/tt-xla/issues/3094"
        )

    run_pooling_test(
        model_name,
        baseline_path,
        max_model_len=64,
        batch_size=batch_size,
        max_num_reqs=batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        optimization_level=optimization_level,
    )


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
