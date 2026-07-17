# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner worker-facing shims.

get_model / reset_mm_cache / add_lora / maybe_setup_dummy_loras /
get_supported_tasks / update_config / reload_weights are pure host logic that
runs on cpu with injected state.
"""

from types import SimpleNamespace

import pytest
from vllm_tt.model_runner_v2 import TTModelRunnerV2


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


@pytest.mark.push
@pytest.mark.cpu
def test_maybe_setup_dummy_loras_noop_context():
    r = object.__new__(TTModelRunnerV2)
    with r.maybe_setup_dummy_loras(None):
        pass  # no-op context must enter/exit cleanly


@pytest.mark.push
@pytest.mark.cpu
def test_maybe_setup_dummy_loras_rejects_lora():
    r = object.__new__(TTModelRunnerV2)
    with pytest.raises(NotImplementedError):
        with r.maybe_setup_dummy_loras(object()):
            pass


@pytest.mark.push
@pytest.mark.cpu
def test_get_supported_tasks_delegates_to_model_state():
    r = object.__new__(TTModelRunnerV2)
    r.model_config = SimpleNamespace(runner_type="generate")
    r.model_state = SimpleNamespace(get_supported_generation_tasks=lambda: ["generate"])
    assert r.get_supported_tasks() == ("generate",)


@pytest.mark.push
@pytest.mark.cpu
def test_get_supported_tasks_rejects_non_generate():
    r = object.__new__(TTModelRunnerV2)
    r.model_config = SimpleNamespace(runner_type="pooling")
    with pytest.raises(NotImplementedError):
        r.get_supported_tasks()


@pytest.mark.push
@pytest.mark.cpu
def test_update_config_rejects_unknown_config():
    r = object.__new__(TTModelRunnerV2)
    with pytest.raises(AssertionError):
        r.update_config({"scheduler_config": {}})


@pytest.mark.push
@pytest.mark.cpu
def test_reload_weights_not_supported():
    r = object.__new__(TTModelRunnerV2)
    with pytest.raises(NotImplementedError):
        r.reload_weights()


@pytest.mark.push
@pytest.mark.cpu
def test_warmup_buckets_cover_all_shapes():
    r = object.__new__(TTModelRunnerV2)
    r.max_num_reqs = 8
    r.min_num_reqs = 2
    r.max_prefill_num_reqs = 4
    r.num_tokens_paddings = [1, 32, 64]
    buckets = r._warmup_buckets()
    # Every distinct request-count crossed with every token padding.
    assert set(buckets) == {
        (t, q) for t in (2, 4, 8) for q in (1, 32, 64)
    }
    assert len(buckets) == 9


@pytest.mark.push
@pytest.mark.cpu
def test_warmup_buckets_dedup_equal_counts():
    r = object.__new__(TTModelRunnerV2)
    r.max_num_reqs = 4
    r.min_num_reqs = 4
    r.max_prefill_num_reqs = 4
    r.num_tokens_paddings = [1, 16]
    buckets = r._warmup_buckets()
    # Collapsed to a single request-count bucket.
    assert buckets == [(4, 1), (4, 16)]
