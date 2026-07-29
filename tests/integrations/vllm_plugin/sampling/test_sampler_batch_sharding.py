# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for data-parallel batch sharding of on-device sampling.

Run via the tt::sampling CPU fallback, so no device is needed. They cover row
bookkeeping and the padding contract, not sampling semantics.
"""

import pytest
import torch
from tt_torch.custom_ops import _sampling_sharding_rule
from vllm_tt.metadata import XLASupportedSamplingMetadata
from vllm_tt.sampler import (
    _TTNN_SAMPLING_BATCH_SIZE,
    Sampler,
    chunked_topk_candidates,
    sampling_pad_rows,
)

VOCAB = 2048


def make_metadata(batch_size, temperature=0.8, top_k=50, top_p=0.9):
    dev = torch.device("cpu")
    return XLASupportedSamplingMetadata(
        temperature=torch.full((batch_size,), temperature, device=dev),
        top_k=torch.full((batch_size,), top_k, dtype=torch.int32, device=dev),
        top_p=torch.full((batch_size,), top_p, device=dev),
        min_p=torch.zeros(batch_size, device=dev),
        all_greedy=False,
    )


# (batch, dp_size) combinations whose local shard fits the kernel.
LEGAL_SHAPES = [
    (1, 1),
    (16, 1),
    (32, 1),
    (64, 2),
    (48, 2),
    (62, 2),
    (16, 2),
    (64, 8),
    (8, 8),
]


@pytest.mark.parametrize("batch,dp_size", LEGAL_SHAPES)
def test_pad_rows_never_truncates(batch, dp_size):
    """The pad amount is never negative, so F.pad can never crop rows."""
    assert sampling_pad_rows(batch, dp_size) >= 0


@pytest.mark.parametrize("batch,dp_size", LEGAL_SHAPES)
def test_padded_batch_is_evenly_shardable(batch, dp_size):
    """Shardy can only split dim 0 if the padded batch divides by dp_size."""
    padded = batch + sampling_pad_rows(batch, dp_size)
    assert padded % dp_size == 0


@pytest.mark.parametrize("batch,dp_size", LEGAL_SHAPES)
def test_local_rows_within_kernel_limit(batch, dp_size):
    """One Tensix core per user caps the kernel at 32 rows per device."""
    padded = batch + sampling_pad_rows(batch, dp_size)
    assert 1 <= padded // dp_size <= _TTNN_SAMPLING_BATCH_SIZE


def test_single_device_still_pads_up_to_kernel_width():
    """dp_size=1 keeps the original behavior: pad small batches up to 32."""
    assert sampling_pad_rows(8, 1) == _TTNN_SAMPLING_BATCH_SIZE - 8
    assert sampling_pad_rows(32, 1) == 0


def test_dp_does_not_pad_to_32_per_replica():
    """A 64-row batch over 8 replicas has 8 local rows, which is legal."""
    assert sampling_pad_rows(64, 8) == 0


@pytest.mark.parametrize("batch,dp_size", [(64, 2), (48, 2), (16, 1), (64, 8)])
def test_sampler_returns_one_token_per_request(batch, dp_size):
    """Every request gets exactly one sampled token."""
    logits = torch.randn(batch, VOCAB)
    sampler = Sampler(dp_size=dp_size)

    out = sampler(logits, make_metadata(batch)).sampled_token_ids

    assert out.shape == (batch, 1), f"expected {batch} rows, got {out.shape}"
    assert out.dtype in (torch.int32, torch.int64)
    assert ((out >= 0) & (out < VOCAB)).all(), "sampled token id out of vocab range"


@pytest.mark.parametrize("batch,dp_size", [(64, 2), (48, 2), (16, 1)])
def test_greedy_and_random_row_counts_agree(batch, dp_size):
    """torch.where in sample() needs both branches to have the same rows."""
    logits = torch.randn(batch, VOCAB)
    sampler = Sampler(dp_size=dp_size)
    metadata = make_metadata(batch)

    greedy = sampler.greedy_sample(logits)
    filtered, indices = chunked_topk_candidates(logits, dp_size)
    random_sampled = sampler._ttnn_sampling_padded(filtered, indices, metadata)

    assert greedy.shape == random_sampled.shape == (batch,)


@pytest.mark.parametrize("batch,dp_size", [(64, 1), (33, 1), (65, 2), (128, 2)])
def test_oversized_local_batch_raises(batch, dp_size):
    """A batch too large for the kernel fails loudly instead of truncating."""
    logits = torch.randn(batch, VOCAB)
    sampler = Sampler(dp_size=dp_size)

    with pytest.raises(ValueError, match="one Tensix core per user"):
        sampler(logits, make_metadata(batch))


def test_chunked_topk_preserves_row_count():
    """chunked_topk_candidates must return the caller's batch, never fewer."""
    for batch, dp_size in LEGAL_SHAPES:
        values, indices = chunked_topk_candidates(torch.randn(batch, VOCAB), dp_size)
        assert values.shape[0] == batch
        assert indices.shape[0] == batch


def test_sharding_rule_ranks_match_operands():
    """tt-mlir hard-errors on a rule whose ranks do not match the operands."""
    rule = _sampling_sharding_rule(
        torch.zeros(64, 256),
        torch.zeros(64, 256, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
        torch.zeros(64),
        torch.zeros(64),
    )

    assert rule == (
        "#sdy.op_sharding_rule<([i, j], [i, j], [i], [i], [i])->([i]) "
        "{i=64, j=256} need_replication={j}, custom>"
    )


def test_sharding_rule_omitted_for_unexpected_ranks():
    """An unrecognized shape yields no rule rather than an invalid one."""
    rule = _sampling_sharding_rule(
        torch.zeros(1, 64, 256),
        torch.zeros(1, 64, 256, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
        torch.zeros(64),
        torch.zeros(64),
    )

    assert rule == ""
