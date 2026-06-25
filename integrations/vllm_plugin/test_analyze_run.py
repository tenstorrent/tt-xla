# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Stdlib-only tests for the offline run analyzer. No hardware/server needed.
#
#   pytest -q integrations/vllm_plugin/test_analyze_run.py

import json
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "tools"))
import analyze_run as ar  # noqa: E402


def _write_run(path):
    """Two requests on a batch-8 server; B's prefill stalls A one step."""
    t = 1000.0
    lines = []

    def add(o):
        lines.append(json.dumps(o))

    def slot(si, rid, state, sched, out):
        return {
            "slot_idx": si,
            "req_id": rid,
            "state": state,
            "num_prompt_tokens": 64,
            "num_computed_tokens": 64,
            "out_len": out,
            "inst_rate": 20,
            "scheduled": sched,
        }

    def snap(ts, slots, kind):
        add(
            {
                "event": "step_snapshot",
                "schema": 2,
                "ts": ts,
                "step_idx": int(ts),
                "num_running": len(slots),
                "num_waiting": 0,
                "num_slots": 8,
                "agg_rate": 0,
                "step_kind": kind,
                "slots": slots,
            }
        )

    add(
        {
            "event": "request_admitted",
            "schema": 2,
            "ts": t,
            "req_id": "A",
            "slot_idx": 0,
            "isl": 64,
        }
    )
    snap(t + 0.1, [slot(0, "A", "PREFILL", 64, 0)], "prefill")
    snap(t + 0.2, [slot(0, "A", "DECODE", 1, 1)], "decode")
    add(
        {
            "event": "request_admitted",
            "schema": 2,
            "ts": t + 0.25,
            "req_id": "B",
            "slot_idx": 1,
            "isl": 64,
        }
    )
    # B prefills -> A present but scheduled 0 == stalled (1 step, ~0.1s)
    snap(
        t + 0.3,
        [slot(0, "A", "DECODE", 0, 1), slot(1, "B", "PREFILL", 64, 0)],
        "prefill",
    )
    snap(
        t + 0.4, [slot(0, "A", "DECODE", 1, 2), slot(1, "B", "DECODE", 1, 1)], "decode"
    )
    add(
        {
            "event": "request_completed",
            "schema": 2,
            "ts": t + 0.5,
            "req_id": "A",
            "slot_idx": 0,
            "isl": 64,
            "out_len": 3,
            "ttft": 0.2,
            "mean_rate": 10.0,
            "finish_reason": "stop",
        }
    )
    add(
        {
            "event": "request_completed",
            "schema": 2,
            "ts": t + 0.6,
            "req_id": "B",
            "slot_idx": 1,
            "isl": 64,
            "out_len": 2,
            "ttft": 0.3,
            "mean_rate": 20.0,
            "finish_reason": "length",
        }
    )
    open(path, "w").write("\n".join(lines) + "\n")


def test_analyzer_metrics(tmp_path):
    ev = str(tmp_path / "events.jsonl")
    _write_run(ev)
    run = ar.build_run(*ar.parse_events(ev))
    m = ar.compute_metrics(run)

    assert m["num_slots"] == 8
    assert m["requests_admitted"] >= 1 and m["requests_completed"] == 2
    assert m["total_output_tokens"] == 5
    assert m["ttft_s"]["n"] == 2 and m["ttft_s"]["max"] == 0.3
    assert m["concurrency"]["max"] == 2
    # B's prefill step stalled A -> some stall time recorded.
    assert m["slot_seconds"]["stalled"] > 0
    assert m["step_kind_seconds"]["prefill"] > 0
    assert m["step_kind_seconds"]["decode"] > 0


def test_analyzer_html_self_contained(tmp_path):
    ev = str(tmp_path / "events.jsonl")
    _write_run(ev)
    run = ar.build_run(*ar.parse_events(ev))
    m = ar.compute_metrics(run)
    out = ar.render_html(run, m, ar.render_markdown(run, m))
    # No external resources; carries the data + an SVG canvas.
    assert "<svg" in out and "const SEGMENTS=" in out
    assert "http" not in out.replace("http://www.w3.org/2000/svg", "")


def test_events_only_degrades(tmp_path):
    """No snapshots -> request bars + overlap, no slot-state breakdown."""
    ev = str(tmp_path / "events.jsonl")
    t = 5.0
    open(ev, "w").write(
        "\n".join(
            json.dumps(o)
            for o in [
                {
                    "event": "request_admitted",
                    "schema": 2,
                    "ts": t,
                    "req_id": "A",
                    "slot_idx": 0,
                    "isl": 32,
                },
                {
                    "event": "request_completed",
                    "schema": 2,
                    "ts": t + 2,
                    "req_id": "A",
                    "slot_idx": 0,
                    "isl": 32,
                    "out_len": 20,
                    "ttft": 0.5,
                    "mean_rate": 10.0,
                    "finish_reason": "stop",
                },
            ]
        )
        + "\n"
    )
    run = ar.build_run(*ar.parse_events(ev))
    assert run["has_snapshots"] is False
    m = ar.compute_metrics(run)
    assert "slot_seconds" not in m  # snapshot-only metric absent
    assert m["osl"]["max"] == 20
    # Still renders an HTML (request-bar mode).
    assert "<svg" in ar.render_html(run, m, ar.render_markdown(run, m))
