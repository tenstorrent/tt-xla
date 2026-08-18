# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the env-gated serving telemetry collectors.

Pure host-side: no TT device, no vLLM engine. Exercises the gating contract
(disabled => zero files, zero I/O), env-var precedence, the buffered-flush
policy (no per-step disk writes), and the record schema for both the scheduler
and runner collectors.

This is the v1-runner branch: the collectors are shared with the v2 runner, so
most of this file is runner-agnostic. The exceptions are covered at the bottom --
``_V1SlotView`` (which adapts v1's condensing ``InputBatch`` to the slot view
the collector reads) and v1's prefix-cache-hit signal (``num_cached_tokens``
rather than ``prefill_len > prompt_len``).
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from vllm_tt.model_runner import _V1SlotView
from vllm_tt.telemetry import (
    RUNNER_FILENAME,
    RUNNER_SNAPSHOT_FILENAME,
    SCHEDULER_FILENAME,
    RunnerTelemetry,
    SchedulerTelemetry,
    reset_sinks,
    resolve_config,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in ("TTXLA_TELEMETRY", "TTXLA_TELEMETRY_DIR", "TTXLA_TELEMETRY_FLUSH_MS"):
        monkeypatch.delenv(var, raising=False)


class _FakeReqStates:
    """Minimal stand-in for TTRequestState with the fields on_step reads."""

    def __init__(self, slots, max_num_reqs=32):
        # slots: {req_id: (slot_idx, num_computed, prefill_len, total_len)}
        self.req_id_to_index = {rid: s[0] for rid, s in slots.items()}
        occupied = set(self.req_id_to_index.values())
        self.free_indices = [i for i in range(max_num_reqs) if i not in occupied]
        self.num_computed_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.prefill_len = np.zeros(max_num_reqs, dtype=np.int32)
        self.total_len = np.zeros(max_num_reqs, dtype=np.int32)
        for _rid, (idx, computed, plen, tlen) in slots.items():
            self.num_computed_tokens[idx] = computed
            self.prefill_len[idx] = plen
            self.total_len[idx] = tlen


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# resolve_config
# --------------------------------------------------------------------------- #
@pytest.mark.push
@pytest.mark.cpu
def test_resolve_config_defaults():
    enabled, directory, flush_s = resolve_config()
    assert enabled is False
    assert directory == "./tt_telemetry"
    assert flush_s == 1.0


@pytest.mark.push
@pytest.mark.cpu
def test_resolve_config_env_overrides_config(monkeypatch, tmp_path):
    monkeypatch.setenv("TTXLA_TELEMETRY", "yes")
    monkeypatch.setenv("TTXLA_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.setenv("TTXLA_TELEMETRY_FLUSH_MS", "250")
    # Config says disabled/elsewhere; env must win.
    enabled, directory, flush_s = resolve_config(
        enabled=False, directory="/other", flush_ms=9999
    )
    assert enabled is True
    assert directory == str(tmp_path)
    assert flush_s == 0.25


@pytest.mark.push
@pytest.mark.cpu
@pytest.mark.parametrize(
    "val,expected", [("1", True), ("on", True), ("0", False), ("off", False)]
)
def test_resolve_config_truthiness(monkeypatch, val, expected):
    monkeypatch.setenv("TTXLA_TELEMETRY", val)
    assert resolve_config()[0] is expected


@pytest.mark.push
@pytest.mark.cpu
def test_resolve_config_bad_flush_ms_falls_back(monkeypatch):
    monkeypatch.setenv("TTXLA_TELEMETRY_FLUSH_MS", "not-a-number")
    assert resolve_config(enabled=True)[2] == 1.0


# --------------------------------------------------------------------------- #
# Disabled => zero-cost, no files
# --------------------------------------------------------------------------- #
@pytest.mark.push
@pytest.mark.cpu
def test_disabled_collectors_write_nothing(tmp_path):
    runner = RunnerTelemetry(False, str(tmp_path), 0.0)
    scheduler = SchedulerTelemetry(False, str(tmp_path), 0.0)
    assert runner.enabled is False and scheduler.enabled is False

    runner.on_request_admitted("r0", 0, 10, 10)
    runner.on_step(
        _FakeReqStates({}), prefill_passes=0, decode_passes=0, emitted_tokens=0
    )
    runner.flush()
    scheduler.on_schedule(
        num_running=1,
        max_running=32,
        num_waiting=0,
        num_free_blocks=10,
        num_total_blocks=10,
        prefill_new=0,
        prefill_resumed=0,
        prefill_partial=0,
        running_scheduled=1,
        preempted=0,
        decode_gated=False,
        decodes_displaced=0,
        total_scheduled_tokens=1,
    )
    scheduler.flush()
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------------------- #
# Buffered flush: no per-step disk I/O
# --------------------------------------------------------------------------- #
@pytest.mark.push
@pytest.mark.cpu
def test_records_buffer_until_flush(tmp_path):
    # Large flush interval => on_step never triggers an interval flush.
    runner = RunnerTelemetry(True, str(tmp_path), flush_s=3600.0)
    rs = _FakeReqStates({"r0": (0, 5, 10, 10)})
    runner.on_step(rs, prefill_passes=1, decode_passes=0, emitted_tokens=0)
    # jsonl exists (truncated on init) but holds no step record yet.
    jsonl = os.path.join(tmp_path, RUNNER_FILENAME)
    assert _read_jsonl(jsonl) == []
    runner.flush()
    assert len(_read_jsonl(jsonl)) == 1


# --------------------------------------------------------------------------- #
# Runner records
# --------------------------------------------------------------------------- #
@pytest.mark.push
@pytest.mark.cpu
def test_runner_admit_and_complete_records(tmp_path):
    runner = RunnerTelemetry(True, str(tmp_path), 0.0)
    runner.on_request_admitted("r0", 3, prompt_len=10, prefill_len=10)
    runner.on_request_admitted("r1", 4, prompt_len=8, prefill_len=12)  # prefix hit
    runner.on_request_completed("r0", 3, prompt_len=10, output_len=20)
    runner.flush()
    recs = _read_jsonl(os.path.join(tmp_path, RUNNER_FILENAME))
    admitted = [r for r in recs if r["event"] == "request_admitted"]
    completed = [r for r in recs if r["event"] == "request_completed"]
    assert {r["request_id"] for r in admitted} == {"r0", "r1"}
    assert (
        next(r for r in admitted if r["request_id"] == "r0")["prefix_cache_hit"]
        is False
    )
    assert (
        next(r for r in admitted if r["request_id"] == "r1")["prefix_cache_hit"] is True
    )
    assert completed[0]["request_id"] == "r0" and completed[0]["output_len"] == 20


@pytest.mark.push
@pytest.mark.cpu
def test_runner_step_occupancy_and_snapshot(tmp_path):
    runner = RunnerTelemetry(True, str(tmp_path), 0.0)
    # r0 still prefilling (computed < prefill_len), r1 decoding.
    rs = _FakeReqStates({"r0": (0, 5, 10, 5), "r1": (1, 12, 12, 15)}, max_num_reqs=32)
    runner.on_step(rs, prefill_passes=1, decode_passes=1, emitted_tokens=1)
    runner.flush()
    step = [
        r
        for r in _read_jsonl(os.path.join(tmp_path, RUNNER_FILENAME))
        if r["event"] == "step"
    ][0]
    assert step["slots_occupied"] == 2
    assert step["slots_free"] == 30
    assert step["num_prefilling"] == 1
    assert step["num_decoding"] == 1
    assert step["prefill_passes"] == 1 and step["decode_passes"] == 1
    assert step["emitted_tokens"] == 1
    # First step has no prior timestamp => decode rate undefined.
    assert step["decode_rate_toks_per_s"] is None

    # Snapshot file holds the latest per-slot picture.
    with open(os.path.join(tmp_path, RUNNER_SNAPSHOT_FILENAME)) as f:
        snap = json.load(f)
    assert {s["request_id"] for s in snap["slots"]} == {"r0", "r1"}
    assert {s["state"] for s in snap["slots"]} == {"PREFILL", "DECODE"}


@pytest.mark.push
@pytest.mark.cpu
def test_runner_decode_rate_after_two_steps(tmp_path):
    runner = RunnerTelemetry(True, str(tmp_path), 0.0)
    rs = _FakeReqStates({"r0": (0, 12, 12, 15)})
    runner.on_step(rs, prefill_passes=0, decode_passes=1, emitted_tokens=1)
    runner.on_step(rs, prefill_passes=0, decode_passes=1, emitted_tokens=1)
    runner.flush()
    steps = [
        r
        for r in _read_jsonl(os.path.join(tmp_path, RUNNER_FILENAME))
        if r["event"] == "step"
    ]
    assert steps[0]["decode_rate_toks_per_s"] is None
    # Second step has a positive dt and one emitted token => positive rate.
    assert steps[1]["decode_rate_toks_per_s"] is not None
    assert steps[1]["decode_rate_toks_per_s"] > 0


# --------------------------------------------------------------------------- #
# Scheduler records
# --------------------------------------------------------------------------- #
@pytest.mark.push
@pytest.mark.cpu
def test_scheduler_utilization_and_cumulative_counters(tmp_path):
    sched = SchedulerTelemetry(True, str(tmp_path), 0.0)
    sched.on_schedule(
        num_running=8,
        max_running=32,
        num_waiting=4,
        num_free_blocks=100,
        num_total_blocks=400,
        prefill_new=2,
        prefill_resumed=0,
        prefill_partial=1,
        running_scheduled=0,
        preempted=1,
        decode_gated=True,
        decodes_displaced=8,
        total_scheduled_tokens=512,
        watermark_rejects=2,
        b1_cap_hit=True,
    )
    sched.on_schedule(
        num_running=8,
        max_running=32,
        num_waiting=2,
        num_free_blocks=300,
        num_total_blocks=400,
        prefill_new=0,
        prefill_resumed=0,
        prefill_partial=0,
        running_scheduled=8,
        preempted=0,
        decode_gated=False,
        decodes_displaced=0,
        total_scheduled_tokens=8,
    )
    sched.flush()
    recs = _read_jsonl(os.path.join(tmp_path, SCHEDULER_FILENAME))
    assert len(recs) == 2
    assert recs[0]["kv_util"] == 0.75  # (400-100)/400
    assert recs[0]["batch_util"] == 0.25  # 8/32
    assert recs[0]["decode_gated"] is True and recs[0]["decodes_displaced"] == 8
    # Cumulative counters accumulate across steps.
    assert recs[1]["cum_preempted"] == 1
    assert recs[1]["cum_decode_stalled_steps"] == 1  # only step 0 was gated
    assert recs[1]["cum_watermark_rejects"] == 2
    assert recs[1]["cum_b1_cap_hits"] == 1


@pytest.mark.push
@pytest.mark.cpu
def test_reset_sinks_clears_all_sinks(tmp_path):
    # A restart must not leave a prior run's data lingering through warmup.
    for name in (SCHEDULER_FILENAME, RUNNER_FILENAME, RUNNER_SNAPSHOT_FILENAME):
        (tmp_path / name).write_text("stale from a previous run")
    reset_sinks(str(tmp_path))
    assert os.listdir(tmp_path) == []
    # Idempotent, and a missing directory is a no-op (best-effort).
    reset_sinks(str(tmp_path))
    reset_sinks(str(tmp_path / "does_not_exist"))


@pytest.mark.push
@pytest.mark.cpu
def test_stale_sink_truncated_on_init(tmp_path):
    path = tmp_path / SCHEDULER_FILENAME
    path.write_text('{"stale":true}\n')
    SchedulerTelemetry(True, str(tmp_path), 0.0)
    # A fresh run must not replay a prior run's records.
    assert path.read_text() == ""


# --------------------------------------------------------------------------- #
# v1-specific: the InputBatch -> slot-view adapter
# --------------------------------------------------------------------------- #
def _fake_v1_state(rows, max_num_reqs=8, scheduled=None):
    """Build (input_batch, requests, scheduler_output) stand-ins for the view.

    ``rows``: {req_id: (row_index, num_computed_tokens, num_prompt_tokens,
    num_output_tokens)}. Mirrors what v1 keeps across InputBatch (row indices)
    and CachedRequestState (token counts).
    """
    input_batch = SimpleNamespace(
        num_reqs=len(rows),
        max_num_reqs=max_num_reqs,
        req_id_to_index={rid: r[0] for rid, r in rows.items()},
    )
    requests = {
        rid: SimpleNamespace(
            num_computed_tokens=computed,
            num_prompt_tokens=prompt,
            num_tokens=prompt + out,
        )
        for rid, (_idx, computed, prompt, out) in rows.items()
    }
    scheduler_output = SimpleNamespace(num_scheduled_tokens=scheduled or {})
    return input_batch, requests, scheduler_output


@pytest.mark.push
@pytest.mark.cpu
def test_v1_slot_view_maps_rows_and_free_tail():
    # Two occupied rows out of 8: v1 has no free list, so the free slots are
    # the tail of the batch rather than an explicit set of indices.
    view = _V1SlotView(
        *_fake_v1_state({"a": (0, 10, 10, 3), "b": (1, 4, 10, 0)}, max_num_reqs=8)
    )
    assert view.req_id_to_index == {"a": 0, "b": 1}
    assert len(view.free_indices) == 6
    assert view.prefill_len[0] == 10 and view.total_len[0] == 13
    assert view.num_computed_tokens[1] == 4


@pytest.mark.push
@pytest.mark.cpu
def test_v1_slot_view_counts_scheduled_tokens_toward_prefill():
    # The view folds in this step's scheduled tokens, so a row finishing its
    # prefill now reports DECODE.
    view = _V1SlotView(*_fake_v1_state({"a": (0, 6, 10, 0)}, scheduled={"a": 4}))
    assert view.num_computed_tokens[0] == 10
    assert view.prefill_len[0] == 10  # => DECODE, not PREFILL


@pytest.mark.push
@pytest.mark.cpu
def test_v1_slot_view_tolerates_row_without_cached_state(tmp_path):
    # A row present in the batch but missing from `requests` must not raise:
    # telemetry never crashes the engine.
    input_batch, requests, sched = _fake_v1_state({"a": (0, 1, 4, 0)})
    requests.pop("a")
    view = _V1SlotView(input_batch, requests, sched)
    assert view.req_id_to_index == {"a": 0}
    # Zeroed rather than dropped: on_step swallows exceptions, so an
    # unpopulated slot would lose the whole step record.
    assert view.prefill_len == {0: 0} and view.num_computed_tokens == {0: 0}
    runner = RunnerTelemetry(True, str(tmp_path), 0.0)
    runner.on_step(view, prefill_passes=0, decode_passes=1, emitted_tokens=1)
    steps = [r for r in _read_jsonl(tmp_path / RUNNER_FILENAME) if r["event"] == "step"]
    assert len(steps) == 1 and steps[0]["slots_occupied"] == 1


@pytest.mark.push
@pytest.mark.cpu
def test_v1_prefix_cache_hit_reported_via_cached_tokens(tmp_path):
    # v2 signals a hit as prefill_len > prompt_len, v1 as advanced
    # num_computed_tokens; both must land as prefix_cache_hit.
    runner = RunnerTelemetry(True, str(tmp_path), 0.0)
    runner.on_request_admitted("miss", 0, 10, 10, num_cached_tokens=0)
    runner.on_request_admitted("hit", 1, 10, 10, num_cached_tokens=6)
    runner.flush()
    admitted = {
        r["request_id"]: r
        for r in _read_jsonl(tmp_path / RUNNER_FILENAME)
        if r["event"] == "request_admitted"
    }
    assert admitted["miss"]["prefix_cache_hit"] is False
    assert admitted["hit"]["prefix_cache_hit"] is True
    assert admitted["hit"]["num_cached_tokens"] == 6


@pytest.mark.push
@pytest.mark.cpu
def test_readmission_is_distinguishable_from_first_admission(tmp_path):
    # Re-admission at a new row with its own prefix cached, so admissions
    # outnumber requests and the hit is not a cross-request one.
    runner = RunnerTelemetry(True, str(tmp_path), 0.0)
    runner.on_request_admitted("r0", 0, 8, 8, num_cached_tokens=0, readmission=False)
    runner.on_request_admitted("r0", 3, 8, 8, num_cached_tokens=8, readmission=True)
    runner.flush()
    adm = [
        r
        for r in _read_jsonl(tmp_path / RUNNER_FILENAME)
        if r["event"] == "request_admitted"
    ]
    assert len(adm) == 2 and len({r["request_id"] for r in adm}) == 1
    first, again = adm
    assert first["readmission"] is False and first["slot"] == 0
    # Same request, different row -- the condensing batch reassigned it.
    assert again["readmission"] is True and again["slot"] == 3
    assert again["prefix_cache_hit"] is True
    # Counting distinct request_ids (or filtering readmission) is what gives 1.
    assert len([r for r in adm if not r["readmission"]]) == 1
