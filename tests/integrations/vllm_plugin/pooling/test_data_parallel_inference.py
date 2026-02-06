# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import vllm

from tests.integrations.vllm_plugin.pooling.utils import run_pooling_test


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.parametrize(
    ["model_name", "baseline_path"],
    [
        pytest.param(
            "BAAI/bge-m3",
            "baseline/bge_m3_baseline.pt",
        ),
    ],
)
@pytest.mark.parametrize(
    "batch_size, max_num_reqs, max_num_batched_tokens",
    [
        (2, 2, 64),
    ],
)
def test_data_parallel_inference_push(
    model_name: str,
    baseline_path: str,
    batch_size: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
):
    """
    Test data parallel inference with vLLM for embedding models.
    """
    run_pooling_test(
        model_name,
        baseline_path,
        max_model_len=64,
        enable_data_parallel=True,
        batch_size=batch_size,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )


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
    "batch_size, max_num_reqs, max_num_batched_tokens",
    [
        (2, 2, 64),
        (4, 4, 128),
    ],
)
def test_data_parallel_inference_nightly(
    model_name: str,
    baseline_path: str,
    batch_size: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
):
    """
    Test data parallel inference with vLLM for embedding models.
    """

    run_pooling_test(
        model_name,
        baseline_path,
        max_model_len=64,
        enable_data_parallel=True,
        batch_size=batch_size,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
