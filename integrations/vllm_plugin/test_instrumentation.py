# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Stdlib-only unit tests for the env-gated telemetry emitter. No hardware,
# server, or vllm import required -- runs anywhere with python3 + pytest.
#
#   pytest -q integrations/vllm_plugin/test_instrumentation.py

import importlib
import json
import os
import sys

# Import the module directly by path so the test doesn't require the vllm_tt
# package (and its vllm deps) to be importable.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "vllm_tt"))
import instrumentation as instr  # noqa: E402


class _FakeSampling:
    temperature = 0.8
    top_k = 20
    top_p = 0.95
    min_p = 0.0
    repetition_penalty = 1.1
    presence_penalty = 0.0
    frequency_penalty = 0.0
    seed = 1234
    max_tokens = 128
    n = 1


class _FakeRequest:
    def __init__(self, req_id, isl):
        self.req_id = req_id
        self.prompt_token_ids = list(range(isl))
        self.sampling_params = _FakeSampling()


class _FakeInputBatch:
    """Minimal stand-in exposing the fields the emitter reads."""

    def __init__(self):
        self.req_ids = ["r0", "r1"]
        self.num_reqs = 2
        self.num_prompt_tokens = [10, 20]
        self.num_computed_tokens_cpu = [10, 5]  # r0 decoding, r1 prefilling
        self.req_output_token_ids = [[1, 2, 3], []]
        self.req_id_to_index = {"r0": 0, "r1": 1}


class _FakeScheduler:
    """Stand-in for scheduler_output: r0 (decode) scheduled out this step,
    r1 (prefill) gets the compute."""

    num_waiting = 2

    def __init__(self):
        self.num_scheduled_tokens = {"r0": 0, "r1": 20}


def _setup(tmp_path, enabled):
    for k in list(os.environ):
        if k.startswith("TT_INSTRUMENT"):
            del os.environ[k]
    if enabled:
        os.environ["TT_INSTRUMENT"] = "1"
        os.environ["TT_INSTRUMENT_DIR"] = str(tmp_path)
        os.environ["TT_INSTRUMENT_THROTTLE_MS"] = "0"  # no throttle in tests
    importlib.reload(instr)
    instr._reset_for_test()
    return str(tmp_path)


def test_disabled_is_noop(tmp_path):
    """With TT_INSTRUMENT unset, nothing is written and enabled() is False."""
    d = _setup(tmp_path, enabled=False)
    assert instr.enabled() is False
    instr.emit_request_admitted(_FakeRequest("r0", 10), 0)
    instr.emit_step_snapshot(_FakeInputBatch())
    instr.emit_request_completed(_FakeInputBatch(), "r0")
    assert os.listdir(d) == [], "disabled emitter must write no files"


def test_enabled_emits_three_event_types(tmp_path):
    d = _setup(tmp_path, enabled=True)
    assert instr.enabled() is True

    instr.emit_request_admitted(_FakeRequest("r0", 10), 0)
    instr.emit_request_admitted(_FakeRequest("r1", 20), 1)
    instr.emit_step_snapshot(_FakeInputBatch())
    instr.emit_request_completed(_FakeInputBatch(), "r0")

    # --- snapshot.json: latest step state ---
    snap_path = os.path.join(d, instr.SNAPSHOT_FILENAME)
    assert os.path.exists(snap_path)
    snap = json.load(open(snap_path))
    assert snap["event"] == instr.EVENT_STEP_SNAPSHOT
    assert snap["schema"] == instr.SCHEMA_VERSION
    assert snap["num_running"] == 2
    states = {s["req_id"]: s["state"] for s in snap["slots"]}
    assert states == {"r0": "DECODE", "r1": "PREFILL"}
    outlens = {s["req_id"]: s["out_len"] for s in snap["slots"]}
    assert outlens == {"r0": 3, "r1": 0}

    # --- events.jsonl: admit + complete ---
    events = [
        json.loads(line)
        for line in open(os.path.join(d, instr.EVENTS_FILENAME))
        if line.strip()
    ]
    kinds = [e["event"] for e in events]
    assert kinds.count(instr.EVENT_REQUEST_ADMITTED) == 2
    assert kinds.count(instr.EVENT_REQUEST_COMPLETED) == 1

    admit = next(e for e in events if e["event"] == instr.EVENT_REQUEST_ADMITTED)
    assert admit["isl"] == 10
    assert admit["sampling"]["temperature"] == 0.8
    assert admit["sampling"]["seed"] == 1234
    assert admit["sampling"]["repetition_penalty"] == 1.1

    done = next(e for e in events if e["event"] == instr.EVENT_REQUEST_COMPLETED)
    assert done["req_id"] == "r0"
    assert done["out_len"] == 3
    assert done["isl"] == 10
    # Enriched completion fields (v2): keys always present.
    for k in ("first_token_ts", "ttft", "mean_rate", "finish_reason"):
        assert k in done


def test_completion_ttft_and_finish_reason(tmp_path):
    """admit -> first decoded token (via snapshot) -> completion yields a ttft
    and a heuristic finish_reason (out_len vs max_tokens)."""
    d = _setup(tmp_path, enabled=True)
    ib = _FakeInputBatch()
    instr.emit_request_admitted(_FakeRequest("r0", 10), 0)  # max_tokens=128
    # r0 has out_len 3 (>0) in the fake batch -> first_token recorded here.
    instr.emit_step_snapshot(ib, _FakeScheduler())
    instr.emit_request_completed(ib, "r0")
    events = [
        json.loads(line)
        for line in open(os.path.join(d, instr.EVENTS_FILENAME))
        if line.strip()
    ]
    done = next(e for e in events if e["event"] == instr.EVENT_REQUEST_COMPLETED)
    assert done["first_token_ts"] is not None
    assert done["ttft"] is not None and done["ttft"] >= 0
    # out_len 3 < max_tokens 128 -> "stop"
    assert done["finish_reason"] == "stop"


def test_step_snapshot_scheduled_and_kind(tmp_path):
    """scheduler_output.num_scheduled_tokens -> per-slot `scheduled` + step_kind,
    the authoritative stall signal."""
    d = _setup(tmp_path, enabled=True)
    instr.emit_step_snapshot(_FakeInputBatch(), _FakeScheduler())
    snap = json.load(open(os.path.join(d, instr.SNAPSHOT_FILENAME)))

    sched = {s["req_id"]: s["scheduled"] for s in snap["slots"]}
    assert sched == {"r0": 0, "r1": 20}  # r0 got no compute -> stalled
    assert snap["num_waiting"] == 2
    # Only r1 (a PREFILL slot) is scheduled, so the step is a prefill step.
    assert snap["step_kind"] == "prefill"


def test_no_scheduler_means_scheduled_none(tmp_path):
    """Without scheduler_output we must not claim scheduled==0 (that would look
    like a stall). It's None == 'unknown'."""
    d = _setup(tmp_path, enabled=True)
    instr.emit_step_snapshot(_FakeInputBatch(), None)
    snap = json.load(open(os.path.join(d, instr.SNAPSHOT_FILENAME)))
    assert all(s["scheduled"] is None for s in snap["slots"])
    assert snap["step_kind"] is None


def test_consumer_derives_stalled(tmp_path):
    """Dashboard maps a DECODE slot with scheduled==0 to STALLED, but leaves it
    DECODE when scheduled is unknown (None)."""
    sys.path.insert(0, os.path.join(_HERE, "tools"))
    import live_dashboard as dash

    m = dash.Model()
    m.upsert_slot(
        {
            "req_id": "r0",
            "slot_idx": 0,
            "state": "DECODE",
            "out_len": 5,
            "num_prompt_tokens": 10,
            "num_computed_tokens": 10,
            "scheduled": 0,
        },
        0.0,
    )
    assert m.streams["r0"].state == dash.S_STALLED

    m.upsert_slot(
        {
            "req_id": "r1",
            "slot_idx": 1,
            "state": "DECODE",
            "out_len": 5,
            "num_prompt_tokens": 10,
            "num_computed_tokens": 10,
            "scheduled": None,
        },
        0.0,
    )
    assert m.streams["r1"].state == dash.S_DECODE


def test_snapshot_throttle(tmp_path):
    """With a throttle window, only the first snapshot in the window writes,
    but the logical step counter still advances."""
    d = _setup(tmp_path, enabled=True)
    os.environ["TT_INSTRUMENT_THROTTLE_MS"] = "100000"  # effectively forever
    importlib.reload(instr)
    instr._reset_for_test()

    ib = _FakeInputBatch()
    instr.emit_step_snapshot(ib)  # first write
    snap1 = json.load(open(os.path.join(d, instr.SNAPSHOT_FILENAME)))
    instr.emit_step_snapshot(ib)  # throttled out
    instr.emit_step_snapshot(ib)  # throttled out
    snap2 = json.load(open(os.path.join(d, instr.SNAPSHOT_FILENAME)))

    assert snap1["step_idx"] == 1
    # File not rewritten -> still the first snapshot, but counter advanced.
    assert snap2["step_idx"] == 1
