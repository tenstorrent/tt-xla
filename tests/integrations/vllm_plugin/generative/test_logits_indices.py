# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for logits_indices computation in batched prefill.

The bug: logits_indices used cumulative token offsets (query_start_loc) to
index into a per-user padded tensor, causing later users' indices to overflow
into earlier users' hidden state slots. This produced cross-user token bleed.

This test is model-free and runs in milliseconds — it directly tests the
index computation logic that select_hidden_states relies on.
"""

import numpy as np
import pytest
import torch


def compute_logits_indices_fixed(num_scheduled_tokens_per_req, max_num_reqs):
    """Current (fixed) implementation."""
    logits_indices = torch.zeros(max_num_reqs, dtype=torch.int32)
    for i, n in enumerate(num_scheduled_tokens_per_req):
        logits_indices[i] = n - 1
    return logits_indices


def compute_logits_indices_buggy(num_scheduled_tokens_per_req, max_num_reqs):
    """Old (buggy) implementation for comparison."""
    query_start_loc = np.zeros(max_num_reqs + 1, dtype=np.int32)
    np.cumsum(
        num_scheduled_tokens_per_req,
        out=query_start_loc[1 : len(num_scheduled_tokens_per_req) + 1],
    )
    query_start_loc[len(num_scheduled_tokens_per_req) + 1 :] = 1
    query_start_loc_cpu = torch.from_numpy(query_start_loc)
    return query_start_loc_cpu[1 : max_num_reqs + 1] - 1


@pytest.mark.parametrize(
    "token_counts,padded_len",
    [
        ([14, 14, 16, 19], 32),  # Original bleed case
        ([15, 17, 14, 14, 16, 19, 16, 17], 32),  # 8-user batch
        ([10, 10, 10, 10], 32),  # Equal lengths
        ([1, 1, 1, 1], 32),  # Minimal
        ([31, 31, 31, 31], 32),  # Near padded_len
        ([14] * 16, 32),  # 16-user batch
    ],
)
def test_logits_indices_within_padded_slot(token_counts, padded_len):
    """Each user's logits_index must be < padded_len (within their own slot)."""
    num_reqs = len(token_counts)
    max_num_reqs = num_reqs
    tokens = np.array(token_counts, dtype=np.int32)

    indices = compute_logits_indices_fixed(tokens, max_num_reqs)

    for i in range(num_reqs):
        assert (
            indices[i] < padded_len
        ), f"User {i}: logits_index {indices[i]} >= padded_len {padded_len}"
        assert (
            indices[i] == token_counts[i] - 1
        ), f"User {i}: logits_index {indices[i]} != expected {token_counts[i] - 1}"


@pytest.mark.parametrize(
    "token_counts,padded_len",
    [
        ([14, 14, 16, 19], 32),
        ([15, 17, 14, 14, 16, 19, 16, 17], 32),
    ],
)
def test_buggy_indices_overflow(token_counts, padded_len):
    """Demonstrate the old buggy implementation produces out-of-bounds indices."""
    num_reqs = len(token_counts)
    tokens = np.array(token_counts, dtype=np.int32)

    buggy = compute_logits_indices_buggy(tokens, num_reqs)
    fixed = compute_logits_indices_fixed(tokens, num_reqs)

    overflow_count = sum(1 for i in range(num_reqs) if buggy[i] >= padded_len)
    assert overflow_count > 0, "Expected buggy impl to overflow but it didn't"

    for i in range(num_reqs):
        assert fixed[i] < padded_len


@pytest.mark.parametrize(
    "token_counts,padded_len",
    [
        ([14, 14, 16, 19], 32),
        ([15, 17, 14, 14, 16, 19, 16, 17], 32),
    ],
)
def test_select_hidden_states_reads_correct_user(token_counts, padded_len):
    """Verify select_hidden_states with fixed indices reads from the correct user."""
    num_reqs = len(token_counts)
    tokens = np.array(token_counts, dtype=np.int32)
    hidden_dim = 8

    # Build fake hidden_states: [num_reqs, padded_len, hidden_dim]
    # Each user's last real position has a unique marker value
    hidden_states = torch.zeros(num_reqs, padded_len, hidden_dim)
    for i in range(num_reqs):
        hidden_states[i, token_counts[i] - 1, :] = float(i + 1)  # marker

    indices = compute_logits_indices_fixed(tokens, num_reqs)
    batch_indices = torch.arange(num_reqs, dtype=torch.int32)
    selected = hidden_states[batch_indices, indices[:num_reqs], :]

    for i in range(num_reqs):
        expected = float(i + 1)
        actual = selected[i, 0].item()
        assert actual == expected, (
            f"User {i}: got marker {actual}, expected {expected} "
            f"(reading from wrong user's slot)"
        )
