# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test for tt-xla #5416: per-batch buffer keying must be consistent.

``TTModelRunner`` preallocates per-batch device buffers (input_ids, position_ids,
logits_indices, batch_idx, page tables) keyed by request-count buckets, and
``_prepare_inputs`` looks them up by a ``target_num_reqs`` chosen per step. The
row count is clamped to the SMEM sequence limit of the active attention path
(``num_reqs_max_model_len`` / ``num_reqs_most_model_len``), which drops below
``max_num_seqs`` at long context.

The bug: input_ids / position_ids / logits_indices / batch_idx were keyed by
``max_num_reqs`` (and ``max_prefill_num_reqs``) while the page tables and the
compiled warmup graphs used the SMEM-clamped ``num_reqs_max_model_len``. When
``num_reqs_max_model_len < max_num_reqs`` the runtime decode row count silently
disagreed with the page tables, and configs using a distinct
``max_prefill_num_reqs`` hit a ``KeyError`` on the buffer lookup.

These tests exercise the pure bucketing helpers (no TT device / model): every
``target_num_reqs`` ``_prepare_inputs`` can request must be a preallocated buffer
key, across the common case and the long-context clamp.
"""

import pytest
from vllm_tt.model_runner import (
    _bucket_num_reqs,
    _reachable_num_reqs,
    _select_target_num_reqs,
)

# Device-free, but the vLLM push matrix only runs device-marked jobs; tag so the
# single_device job collects these (they need no device and run in milliseconds).
pytestmark = [pytest.mark.push, pytest.mark.single_device]


def _all_reachable_targets(
    min_num_reqs,
    max_prefill_num_reqs,
    num_reqs_max_model_len,
    num_reqs_most_model_len,
):
    """Every ``target_num_reqs`` ``_prepare_inputs`` can produce for a config.

    Enumerates both attention paths, decode + prefill, and every possible
    ``actual_num_reqs`` (already truncated to the path limit upstream).
    """
    targets = set()
    paths = [num_reqs_max_model_len]
    if num_reqs_most_model_len is not None:
        paths.append(num_reqs_most_model_len)
    for path_max in paths:
        for is_decode in (True, False):
            for actual in range(1, path_max + 1):
                targets.add(
                    _select_target_num_reqs(
                        min_num_reqs,
                        max_prefill_num_reqs,
                        path_max,
                        is_decode,
                        actual,
                    )
                )
    return targets


# (min_num_reqs, max_prefill_num_reqs, max_num_reqs, N_max, N_most)
# Invariants: N_max <= N_most <= max_num_reqs; max_prefill_num_reqs <= max_num_reqs.
_CONFIGS = [
    # Common case: SMEM limit >= max_num_seqs, so no clamp (all counts equal).
    (32, 32, 32, 32, None),
    (32, 32, 32, 32, 32),
    # min_num_seqs feature: distinct small prefill bucket, no clamp.
    (4, 32, 32, 32, 32),
    # max_prefill feature: prefill capped below decode, no clamp.
    (4, 16, 32, 32, None),
    # Long context: SMEM limit below max_num_seqs (the #5416 trigger), feature off.
    (32, 32, 32, 16, None),
    (32, 32, 32, 8, 16),
    # Long context + min_num_seqs feature.
    (4, 32, 32, 16, 24),
    # Long context + max_prefill below the SMEM limit (three distinct buckets).
    (4, 16, 32, 24, None),
    # Long context clamps the prefill bucket too (max_prefill > SMEM limit).
    (4, 16, 32, 8, None),
    # Extreme clamp: only one sequence fits in SMEM.
    (32, 32, 32, 1, None),
]


@pytest.mark.parametrize(
    "min_num_reqs,max_prefill_num_reqs,max_num_reqs,num_reqs_max_model_len,"
    "num_reqs_most_model_len",
    _CONFIGS,
)
def test_every_target_num_reqs_has_a_buffer_key(
    min_num_reqs,
    max_prefill_num_reqs,
    max_num_reqs,
    num_reqs_max_model_len,
    num_reqs_most_model_len,
):
    """Buffers keyed by ``_reachable_num_reqs`` cover every requestable target."""
    keys = _reachable_num_reqs(
        min_num_reqs,
        max_prefill_num_reqs,
        num_reqs_max_model_len,
        num_reqs_most_model_len,
    )
    targets = _all_reachable_targets(
        min_num_reqs,
        max_prefill_num_reqs,
        num_reqs_max_model_len,
        num_reqs_most_model_len,
    )
    missing = targets - keys
    assert not missing, (
        f"target_num_reqs {sorted(missing)} would miss the buffer dict keyed by "
        f"{sorted(keys)}"
    )


@pytest.mark.parametrize(
    "min_num_reqs,max_prefill_num_reqs,max_num_reqs,num_reqs_max_model_len,"
    "num_reqs_most_model_len",
    _CONFIGS,
)
def test_reachable_key_set_has_no_unused_keys(
    min_num_reqs,
    max_prefill_num_reqs,
    max_num_reqs,
    num_reqs_max_model_len,
    num_reqs_most_model_len,
):
    """No allocated buffer key is unreachable, so the key set stays minimal."""
    keys = _reachable_num_reqs(
        min_num_reqs,
        max_prefill_num_reqs,
        num_reqs_max_model_len,
        num_reqs_most_model_len,
    )
    targets = _all_reachable_targets(
        min_num_reqs,
        max_prefill_num_reqs,
        num_reqs_max_model_len,
        num_reqs_most_model_len,
    )
    assert keys == targets


def test_decode_target_is_clamped_to_smem_limit():
    """Decode targets the SMEM limit, not max_num_reqs, once the clamp is active.

    This is the #5416 fix: pre-fix, decode requested ``max_num_reqs`` while the
    page tables and warmup used ``num_reqs_max_model_len``.
    """
    min_num_reqs, max_prefill_num_reqs, num_reqs_max_model_len = 32, 32, 16
    decode_target = _select_target_num_reqs(
        min_num_reqs,
        max_prefill_num_reqs,
        num_reqs_max_model_len,
        is_decode_step=True,
        actual_num_reqs=1,
    )
    assert decode_target == num_reqs_max_model_len
    assert decode_target in _reachable_num_reqs(
        min_num_reqs, max_prefill_num_reqs, num_reqs_max_model_len, None
    )


def test_buckets_never_exceed_smem_limit():
    """Every bucket stays within the path's SMEM seq limit even when
    ``min_num_reqs`` / ``max_prefill_num_reqs`` are larger."""
    small, big, decode = _bucket_num_reqs(
        min_num_reqs=32, max_prefill_num_reqs=32, path_max_num_reqs=16
    )
    assert (small, big, decode) == (16, 16, 16)
    small, big, decode = _bucket_num_reqs(
        min_num_reqs=4, max_prefill_num_reqs=24, path_max_num_reqs=16
    )
    assert (small, big, decode) == (4, 16, 16)
    small, big, decode = _bucket_num_reqs(
        min_num_reqs=4, max_prefill_num_reqs=8, path_max_num_reqs=16
    )
    assert (small, big, decode) == (4, 8, 16)
