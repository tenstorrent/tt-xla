# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test for tt-xla #5416: per-batch buffer keying must be consistent.

``TTModelRunner`` preallocates per-batch device buffers keyed by request-count
buckets; ``_prepare_inputs`` looks them up by a per-step ``target_num_reqs``
clamped to the active path's SMEM seq limit (``num_reqs_max_model_len``),
which drops below ``max_num_seqs`` at long context.
Pre-fix the shared buffers were keyed by ``max_num_reqs`` while the page tables
and warmup used the clamped count, so decode ran at the wrong row count and a
distinct ``max_prefill_num_reqs`` hit a ``KeyError``.

These tests exercise the pure bucketing helpers (no device): every requestable
``target_num_reqs`` must be a preallocated buffer key.
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
):
    """Every ``target_num_reqs`` ``_prepare_inputs`` can produce for a config.

    Enumerates decode + prefill, and every possible ``actual_num_reqs``
    (already truncated to the path limit upstream).
    """
    targets = set()

    for is_decode in (True, False):
        for actual in range(1, num_reqs_max_model_len + 1):
            targets.add(
                _select_target_num_reqs(
                    min_num_reqs,
                    max_prefill_num_reqs,
                    num_reqs_max_model_len,
                    is_decode,
                    actual,
                )
            )
    return targets


# (min_num_reqs, max_prefill_num_reqs, max_num_reqs, N_max)
# Invariants: max_prefill_num_reqs <= max_num_reqs.
_CONFIGS = [
    # Common case: SMEM limit >= max_num_seqs, so no clamp.
    (32, 32, 32, 32),
    # min_num_seqs feature: distinct small prefill bucket, no clamp.
    (4, 32, 32, 32),
    # max_prefill feature: prefill capped below decode, no clamp.
    (4, 16, 32, 32),
    # Long context: SMEM limit below max_num_seqs (the #5416 trigger).
    (32, 32, 32, 16),
    # Long context + min_num_seqs feature.
    (4, 32, 32, 16),
    # Long context + max_prefill below the SMEM limit (three distinct buckets).
    (4, 16, 32, 24),
    # Long context clamps the prefill bucket too (max_prefill > SMEM limit).
    (4, 16, 32, 8),
    # Extreme clamp: only one sequence fits in SMEM.
    (32, 32, 32, 1),
]


@pytest.mark.parametrize(
    "min_num_reqs,max_prefill_num_reqs,max_num_reqs,num_reqs_max_model_len",
    _CONFIGS,
)
def test_every_target_num_reqs_has_a_buffer_key(
    min_num_reqs,
    max_prefill_num_reqs,
    max_num_reqs,
    num_reqs_max_model_len,
):
    """Buffers keyed by ``_reachable_num_reqs`` cover every requestable target."""
    keys = _reachable_num_reqs(
        min_num_reqs,
        max_prefill_num_reqs,
        num_reqs_max_model_len,
    )
    targets = _all_reachable_targets(
        min_num_reqs,
        max_prefill_num_reqs,
        num_reqs_max_model_len,
    )
    missing = targets - keys
    assert not missing, (
        f"target_num_reqs {sorted(missing)} would miss the buffer dict keyed by "
        f"{sorted(keys)}"
    )


@pytest.mark.parametrize(
    "min_num_reqs,max_prefill_num_reqs,max_num_reqs,num_reqs_max_model_len",
    _CONFIGS,
)
def test_reachable_key_set_has_no_unused_keys(
    min_num_reqs,
    max_prefill_num_reqs,
    max_num_reqs,
    num_reqs_max_model_len,
):
    """No allocated buffer key is unreachable, so the key set stays minimal."""
    keys = _reachable_num_reqs(
        min_num_reqs,
        max_prefill_num_reqs,
        num_reqs_max_model_len,
    )
    targets = _all_reachable_targets(
        min_num_reqs,
        max_prefill_num_reqs,
        num_reqs_max_model_len,
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
        min_num_reqs, max_prefill_num_reqs, num_reqs_max_model_len
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
