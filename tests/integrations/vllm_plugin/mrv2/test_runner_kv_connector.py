# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MRv2 runner's KV-connector (disaggregated serving) wiring.

Three seams, all no-ops when no transfer group is configured:
* ``initialize_kv_cache`` registers the allocated caches with the connector.
* ``execute_model`` drives send/recv even on a step that schedules no tokens.
* ``sample_tokens`` opens the connector lifecycle ONCE per step, around the whole
  multi-pass loop, and attaches the resulting output.

The once-per-step property is the load-bearing one: ``get_finished()`` reports a
transfer only on its first call, so entering per pass would drop finished
send/recv ids on every pass after the first.
"""

import contextlib
from types import SimpleNamespace

import pytest
from test_runner_driver import make_runner, make_sched
from vllm_tt.model_runner_v2 import TTModelRunnerV2


class FakeConnector:
    def __init__(self):
        self.registered = None
        self.xfer_ops = None

    def register_kv_caches(self, kv_caches):
        self.registered = kv_caches

    def set_host_xfer_buffer_ops(self, fn):
        self.xfer_ops = fn


@pytest.fixture
def connector(monkeypatch):
    """Install a fake transfer group; yields it plus an enter-counting context."""
    fake = FakeConnector()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.has_kv_transfer_group", lambda: True
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.get_kv_transfer_group", lambda: fake
    )
    return fake


@pytest.mark.push
@pytest.mark.cpu
def test_register_kv_caches_noop_without_transfer_group(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.has_kv_transfer_group", lambda: False
    )
    r = object.__new__(TTModelRunnerV2)
    # Must not raise, and must not touch get_kv_transfer_group.
    r._register_kv_caches_for_transfer({"layer0": object()})


@pytest.mark.push
@pytest.mark.cpu
def test_register_kv_caches_hands_caches_to_connector(connector):
    r = object.__new__(TTModelRunnerV2)
    caches = {"layer0": object(), "layer1": object()}
    r._register_kv_caches_for_transfer(caches)
    assert connector.registered is caches
    assert connector.xfer_ops is not None


@pytest.mark.push
@pytest.mark.cpu
def test_no_token_step_drives_connector_instead_of_empty_output(monkeypatch):
    r = make_runner()
    r.vllm_config = SimpleNamespace()
    monkeypatch.setattr(r, "_has_kv_transfer_group", lambda: True, raising=False)
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.has_kv_transfer_group", lambda: True
    )
    sentinel = SimpleNamespace(req_ids=["sentinel"])
    monkeypatch.setattr(
        TTModelRunnerV2,
        "kv_connector_no_forward",
        staticmethod(lambda so, cfg: sentinel),
    )
    out = r.execute_model(make_sched(num_sched={}, total=0))
    assert out is sentinel
    assert r.scheduler_output is None  # nothing stashed


@pytest.mark.push
@pytest.mark.cpu
def test_no_token_step_returns_empty_output_without_connector():
    r = make_runner()
    out = r.execute_model(make_sched(num_sched={}, total=0))
    assert out.req_ids == []


@pytest.mark.push
@pytest.mark.cpu
def test_sample_tokens_opens_connector_once_around_the_whole_pass_loop(monkeypatch):
    r = make_runner()
    r.vllm_config = SimpleNamespace()
    r.scheduler_output = make_sched(num_sched={}, total=0)

    enters = []
    active = []
    kv_out = SimpleNamespace(finished_sending=None)

    @contextlib.contextmanager
    def fake_connector_output(scheduler_output, defer_finalize=False):
        enters.append(scheduler_output)
        active.append(True)
        try:
            yield kv_out
        finally:
            active.pop()

    monkeypatch.setattr(r, "_has_kv_transfer_group", lambda: True, raising=False)
    monkeypatch.setattr(
        r, "maybe_get_kv_connector_output", fake_connector_output, raising=False
    )
    # The runner pushes a token-less forward context so start_load_kv has one.
    monkeypatch.setattr(
        "vllm.forward_context.set_forward_context",
        lambda *a, **k: contextlib.nullcontext(),
    )

    # Several passes' worth of work, but the connector must be entered once.
    loop_calls = []

    def fake_loop(*a, **k):
        loop_calls.append(bool(active))  # connector active while passes run?
        loop_calls.append(bool(active))

    monkeypatch.setattr(r, "_run_pass_loop", fake_loop, raising=False)
    monkeypatch.setattr(r, "_get_prompt_logprobs_dict", lambda *a, **k: {})

    out = r.sample_tokens(None)

    assert len(enters) == 1, "connector lifecycle must open once per step"
    assert all(loop_calls), "passes must run inside the connector window"
    assert out.kv_connector_output is kv_out


@pytest.mark.push
@pytest.mark.cpu
def test_sample_tokens_leaves_connector_output_none_without_group(monkeypatch):
    r = make_runner()
    r.scheduler_output = make_sched(num_sched={}, total=0)
    monkeypatch.setattr(r, "_run_pass_loop", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(r, "_get_prompt_logprobs_dict", lambda *a, **k: {})

    out = r.sample_tokens(None)
    assert out.kv_connector_output is None
