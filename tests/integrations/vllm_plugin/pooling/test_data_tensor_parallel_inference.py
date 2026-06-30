# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Data-parallel + tensor-parallel (DP+TP) inference tests for pooling models.

DP+TP builds an SPMD mesh (dp_size, tp_size). The input batch is sharded on the
"batch" (DP) axis; weights are sharded on the "model" (TP) axis, plus the
"batch" axis when shard_weights_on_batch_axis=True (FSDP-style). Embeddings are
checked against the single-chip baseline at PCC >= 0.99.
"""
import pytest

from tests.integrations.vllm_plugin.pooling.utils import run_pooling_test


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "baseline_path"],
    [
        pytest.param(
            "intfloat/e5-mistral-7b-instruct",
            "baseline/e5_mistral_7b_instruct_baseline.pt",
        ),
    ],
)
@pytest.mark.parametrize(
    "max_num_reqs, max_num_batched_tokens",
    [
        (2, 64),  # tight: 1 sentence per replica
        (4, 128),  # wider: 2 sentences per replica
    ],
)
@pytest.mark.parametrize("shard_weights_on_batch_axis", [True, False])
def test_data_tensor_parallel_inference_push(
    model_name: str,
    baseline_path: str,
    max_num_reqs: int,
    max_num_batched_tokens: int,
    shard_weights_on_batch_axis: bool,
):
    """Pooling DP+TP smoke test on llmbox (8 chips → mesh (2, 4))."""
    run_pooling_test(
        model_name,
        baseline_path,
        max_model_len=64,
        enable_tensor_parallel=True,
        enable_data_parallel=True,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
        shard_weights_on_batch_axis=shard_weights_on_batch_axis,
    )


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "baseline_path"],
    [
        # BGE-M3 omitted: small enough that DP-only suffices (see
        # test_data_parallel_inference.py).
        pytest.param(
            "Qwen/Qwen3-Embedding-4B",
            "baseline/qwen3_embedding_4B_baseline.pt",
        ),
    ],
)
@pytest.mark.parametrize(
    "max_num_reqs, max_num_batched_tokens",
    [
        (2, 64),
        (4, 128),
    ],
)
def test_data_tensor_parallel_inference_nightly(
    model_name: str,
    baseline_path: str,
    max_num_reqs: int,
    max_num_batched_tokens: int,
):
    """Pooling DP+TP nightly: larger model (4B) plus the bge-m3 baseline."""
    run_pooling_test(
        model_name,
        baseline_path,
        max_model_len=64,
        enable_tensor_parallel=True,
        enable_data_parallel=True,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
