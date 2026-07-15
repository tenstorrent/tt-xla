# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import vllm

from tests.integrations.vllm_plugin.pooling.utils import run_pooling_test


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.dual_chip
@pytest.mark.parametrize(
    ["model_name", "baseline_path"],
    [
        pytest.param(
            "intfloat/e5-mistral-7b-instruct",
            "baseline/e5_mistral_7b_instruct_baseline.pt",
        ),
    ],
)
def test_tensor_parallel_n300(
    model_name: str,
    baseline_path: str,
):
    """
    Test tensor parallel inference with vLLM for embedding models on N300.
    """

    run_pooling_test(
        model_name,
        baseline_path,
        max_model_len=64,
        enable_tensor_parallel=True,
        max_num_batched_tokens=128,
    )


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "baseline_path"],
    [
        pytest.param(
            "Qwen/Qwen3-Embedding-4B",
            "baseline/qwen3_embedding_4B_baseline.pt",
        ),
        pytest.param(
            "Qwen/Qwen3-Embedding-8B",
            "baseline/qwen3_embedding_8B_baseline.pt",
        ),
    ],
)
def test_tensor_parallel_llmbox(
    model_name: str,
    baseline_path: str,
):
    """
    Test tensor parallel inference with vLLM for embedding models on LLMBox.
    """

    run_pooling_test(
        model_name,
        baseline_path,
        max_model_len=64,
        enable_tensor_parallel=True,
        max_num_batched_tokens=128,
    )
