# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner warmup buckets + worker shims.

``TTModelRunnerV2._warmup_buckets`` / ``get_model`` / ``reset_mm_cache`` /
``add_lora`` (see vllm_tt/model_runner_v2.py) are pure host logic; the actual
``capture_model`` precompile forward needs a model + TT device and runs at
stand-up. Here the bucket enumeration and the shims run on cpu with injected
state.

They pin the warmup shape coverage (decode + prefill request buckets x every
token-length padding, deduplicated) and the worker-facing shim behavior.
"""

import pytest

from vllm_tt.model_runner_v2 import TTModelRunnerV2


def make_runner(
    max_num_reqs=8, min_num_reqs=2, max_prefill_num_reqs=4, paddings=(1, 32)
):
    r = object.__new__(TTModelRunnerV2)
    r.max_num_reqs = max_num_reqs
    r.min_num_reqs = min_num_reqs
    r.max_prefill_num_reqs = max_prefill_num_reqs
    r.num_tokens_paddings = list(paddings)
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_warmup_buckets_cover_request_and_token_shapes():
    r = make_runner()
    # targets = sorted({8, 2, 4}) = [2, 4, 8]; each x every token padding.
    assert r._warmup_buckets() == [(2, 1), (2, 32), (4, 1), (4, 32), (8, 1), (8, 32)]


@pytest.mark.push
@pytest.mark.cpu
def test_warmup_buckets_dedup_when_request_counts_equal():
    r = make_runner(
        max_num_reqs=8, min_num_reqs=8, max_prefill_num_reqs=8, paddings=(1, 16)
    )
    # All request-count buckets coincide -> a single target.
    assert r._warmup_buckets() == [(8, 1), (8, 16)]


@pytest.mark.push
@pytest.mark.cpu
def test_get_model_returns_the_model():
    r = object.__new__(TTModelRunnerV2)
    sentinel = object()
    r.model = sentinel
    assert r.get_model() is sentinel


@pytest.mark.push
@pytest.mark.cpu
def test_reset_mm_cache_noop_without_budget():
    r = object.__new__(TTModelRunnerV2)
    r.mm_budget = None
    r.reset_mm_cache()  # must not raise


@pytest.mark.push
@pytest.mark.cpu
def test_reset_mm_cache_resets_budget():
    r = object.__new__(TTModelRunnerV2)
    calls = []
    r.mm_budget = type("B", (), {"reset_cache": lambda self: calls.append(1)})()
    r.reset_mm_cache()
    assert calls == [1]


@pytest.mark.push
@pytest.mark.cpu
def test_add_lora_not_supported():
    r = object.__new__(TTModelRunnerV2)
    with pytest.raises(NotImplementedError):
        r.add_lora(object())
