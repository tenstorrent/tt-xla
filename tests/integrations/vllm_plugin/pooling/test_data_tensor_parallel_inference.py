# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Combined data-parallel + tensor-parallel (DP+TP) inference tests for
pooling/embedding models.

Background
----------
With both `enable_data_parallel=True` and `enable_tensor_parallel=True` the
runner builds an SPMD mesh of shape ``(dp_size, tp_size)``:

  * On an 8-chip llmbox: ``(2, 4)`` — 2 DP replicas, each running 4-way TP.
  * On a 4-chip box:     ``(2, 2)`` — 2 DP replicas, each running 2-way TP.

Weight sharding depends on ``shard_weights_on_batch_axis``:
  * ``False`` (classic DP+TP): weights are sharded only along the ``"model"``
    (TP) axis; DP replicas hold identical weight slices and never communicate.
  * ``True`` (FSDP-style): weights are additionally sharded along the
    ``"batch"`` (DP) axis, trading extra collectives for lower per-device
    weight memory.
The input batch is always sharded along the ``"batch"`` (DP) axis, so each
replica sees a disjoint subset of the input sentences.

What this test checks
---------------------
1. The engine builds, the model loads, and ``shard_model()`` runs without
   crashing under the new ``(dp_size, tp_size)`` mesh.
2. Embeddings round-trip through DP+TP execution and match the single-chip
   baseline within PCC ≥ 0.99 (the same threshold used by the DP-only and
   TP-only tests).
3. The runtime ``_prepare_inputs`` row-padding logic uses ``dp_size`` (not
   ``num_devices``) so a batch of 4 sentences fits in 1 step on mesh
   ``(2, 4)`` (4 % 2 == 0, no padding rows added).
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
        # Tight fit: with 8 chips → mesh (2, 4), dp_size=2, so max_num_reqs=2
        # means each DP replica handles 1 sentence per step.
        (2, 64),
        # Wider batch: each replica handles 2 sentences per step. Exercises
        # the per-replica batching path in addition to plain shard-split.
        (4, 128),
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
        pytest.param(
            "BAAI/bge-m3",
            "baseline/bge_m3_baseline.pt",
            marks=pytest.mark.xfail(
                reason="BGE-M3 / XLMRoberta produces malformed sdy.all_slice "
                "under any model-axis sharding (TP-only or DP+TP). "
                "Pre-existing bug in vllm_distributed_utils.shard_model(); "
                "tracked separately.",
                strict=False,
            ),
        ),
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
