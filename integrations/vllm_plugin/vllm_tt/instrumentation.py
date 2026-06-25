# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Env-gated, hot-path-safe telemetry for the vLLM TT engine.
#
# This module is the shared CONTRACT between the engine (producer) and any
# viewer (consumer, e.g. tools/live_dashboard.py). It emits three event types
# and writes two sinks; the schema + file layout below is the interface, not
# the TUI.
#
# Hard rules (see handoff / lesson from #4278):
#   - STDLIB ONLY. This runs inside the EngineCore hot path; it must never pull
#     in textual/rich/httpx or anything heavy.
#   - When disabled (TT_INSTRUMENT unset/falsey) every entry point is a cheap
#     guarded no-op: a single cached bool check, no allocation, no I/O.
#   - Per-step work is strictly O(active_slots), never O(output_len). The step
#     snapshot is a fixed-width-per-slot dict; admit/complete are O(1). The
#     snapshot is additionally throttled (TT_INSTRUMENT_THROTTLE_MS).
#
# Env vars:
#   TT_INSTRUMENT          truthy ("1","true","yes","on") enables emission
#   TT_INSTRUMENT_DIR      sink directory (default: ./.tt_instrument)
#   TT_INSTRUMENT_THROTTLE_MS  min gap between step snapshots (default: 50)
#   TT_INSTRUMENT_EVENTS   "1" (default) also append admit/complete to events.jsonl
#
# Sinks (both live in TT_INSTRUMENT_DIR):
#   snapshot.json   latest step_snapshot only, atomically overwritten. A live
#                   viewer tails/polls this for *current* state.
#   events.jsonl    append-only log of request_admitted + request_completed
#                   (and, if TT_INSTRUMENT_SNAPSHOTS_JSONL=1, snapshots too).
#                   For offline analysis / request-mix audits.

import json
import os
import time
from typing import Any, Optional

# Schema version: bump when the event/snapshot field layout changes so consumers
# can detect mismatches.
SCHEMA_VERSION = 1

EVENT_REQUEST_ADMITTED = "request_admitted"
EVENT_STEP_SNAPSHOT = "step_snapshot"
EVENT_REQUEST_COMPLETED = "request_completed"

SNAPSHOT_FILENAME = "snapshot.json"
EVENTS_FILENAME = "events.jsonl"

_TRUTHY = {"1", "true", "yes", "on"}


class _State:
    """Lazily-initialized module state. Kept in one object so the disabled
    path is a single attribute read."""

    def __init__(self) -> None:
        self.enabled: Optional[bool] = None  # tri-state: None == not yet resolved
        self.dir: str = ""
        self.snapshot_path: str = ""
        self.events_path: str = ""
        self.throttle_s: float = 0.05
        self.write_events: bool = True
        self.snapshots_to_jsonl: bool = False
        # Throttle bookkeeping for step snapshots.
        self.last_snapshot_t: float = 0.0
        self.step_idx: int = 0
        # Per-request decode-rate tracking: req_id -> (last_out_len, last_ts).
        # Bounded by active slots; entries are dropped on completion.
        self.rate_track: dict[str, tuple[int, float]] = {}


_S = _State()


def _resolve() -> bool:
    """Resolve env config once. Returns whether instrumentation is enabled."""
    if _S.enabled is not None:
        return _S.enabled

    val = os.environ.get("TT_INSTRUMENT", "").strip().lower()
    _S.enabled = val in _TRUTHY
    if not _S.enabled:
        return False

    _S.dir = os.environ.get("TT_INSTRUMENT_DIR", "").strip() or os.path.join(
        os.getcwd(), ".tt_instrument"
    )
    try:
        os.makedirs(_S.dir, exist_ok=True)
    except OSError:
        # If we can't create the sink dir, disable rather than crash the engine.
        _S.enabled = False
        return False

    _S.snapshot_path = os.path.join(_S.dir, SNAPSHOT_FILENAME)
    _S.events_path = os.path.join(_S.dir, EVENTS_FILENAME)

    try:
        throttle_ms = float(os.environ.get("TT_INSTRUMENT_THROTTLE_MS", "50"))
    except ValueError:
        throttle_ms = 50.0
    _S.throttle_s = max(0.0, throttle_ms / 1000.0)

    _S.write_events = (
        os.environ.get("TT_INSTRUMENT_EVENTS", "1").strip().lower() in _TRUTHY
    )
    _S.snapshots_to_jsonl = (
        os.environ.get("TT_INSTRUMENT_SNAPSHOTS_JSONL", "0").strip().lower() in _TRUTHY
    )

    # Fresh run: truncate the events log so a viewer doesn't replay a stale run.
    try:
        open(_S.events_path, "w").close()
    except OSError:
        pass
    return True


def enabled() -> bool:
    """Public, cheap gate. Call this before doing any work in a hook."""
    e = _S.enabled
    if e is None:
        return _resolve()
    return e


def _append_event(obj: dict[str, Any]) -> None:
    """Append one JSON object as a line to events.jsonl. Best-effort."""
    try:
        line = json.dumps(obj, separators=(",", ":"))
        with open(_S.events_path, "a") as f:
            f.write(line + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _write_snapshot(obj: dict[str, Any]) -> None:
    """Atomically overwrite snapshot.json (write temp + os.replace) so a
    reader never sees a half-written file."""
    try:
        tmp = _S.snapshot_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        os.replace(tmp, _S.snapshot_path)
    except (OSError, TypeError, ValueError):
        pass


def _sampling_summary(sampling_params: Any) -> dict[str, Any]:
    """Pull the interesting sampling fields off a vllm SamplingParams without
    importing vllm. All accesses are guarded; missing fields become None."""

    def g(name: str) -> Any:
        return getattr(sampling_params, name, None)

    return {
        "temperature": g("temperature"),
        "top_k": g("top_k"),
        "top_p": g("top_p"),
        "min_p": g("min_p"),
        "repetition_penalty": g("repetition_penalty"),
        "presence_penalty": g("presence_penalty"),
        "frequency_penalty": g("frequency_penalty"),
        "seed": g("seed"),
        "max_tokens": g("max_tokens"),
        "n": g("n"),
    }


# --------------------------------------------------------------------------- #
# Event 1: request_admitted
# --------------------------------------------------------------------------- #
def emit_request_admitted(request: Any, req_index: Optional[int] = None) -> None:
    """Hook: InputBatch.add_request(). `request` is a CachedRequestState.

    O(1) per request. Records arrival, ISL, sampling params and slot index.
    """
    if not enabled():
        return
    try:
        req_id = getattr(request, "req_id", None)
        prompt_ids = getattr(request, "prompt_token_ids", None)
        isl = len(prompt_ids) if prompt_ids is not None else None
        sampling = getattr(request, "sampling_params", None)
        now = time.time()
        # Seed decode-rate tracking so the first snapshot has a baseline.
        if req_id is not None:
            _S.rate_track[req_id] = (0, now)
        evt = {
            "schema": SCHEMA_VERSION,
            "event": EVENT_REQUEST_ADMITTED,
            "ts": now,
            "req_id": req_id,
            "slot_idx": req_index,
            "isl": isl,
            "sampling": _sampling_summary(sampling) if sampling is not None else None,
        }
        if _S.write_events:
            _append_event(evt)
    except Exception:
        # Telemetry must never take down the engine.
        pass


# --------------------------------------------------------------------------- #
# Event 2: step_snapshot
# --------------------------------------------------------------------------- #
def emit_step_snapshot(input_batch: Any, scheduler_output: Any = None) -> None:
    """Hook: model_runner execute step, after _update_states().

    Builds the current per-slot picture: O(active_slots). Throttled to
    TT_INSTRUMENT_THROTTLE_MS so a fast decode loop doesn't spam the disk.
    """
    if not enabled():
        return
    try:
        now = time.time()
        # Always bump the logical step counter, but throttle disk writes.
        _S.step_idx += 1
        if (now - _S.last_snapshot_t) < _S.throttle_s:
            return
        _S.last_snapshot_t = now

        num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
        req_ids = getattr(input_batch, "req_ids", []) or []
        num_computed = getattr(input_batch, "num_computed_tokens_cpu", None)
        num_prompt = getattr(input_batch, "num_prompt_tokens", None)
        out_token_ids = getattr(input_batch, "req_output_token_ids", None)

        slots = []
        agg_rate = 0.0
        live_ids = set()
        for i in range(num_reqs):
            req_id = req_ids[i] if i < len(req_ids) else None
            if req_id is None:
                continue
            live_ids.add(req_id)
            n_prompt = int(num_prompt[i]) if num_prompt is not None else None
            n_computed = int(num_computed[i]) if num_computed is not None else None
            out_len = 0
            if out_token_ids is not None and i < len(out_token_ids):
                ot = out_token_ids[i]
                out_len = len(ot) if ot is not None else 0

            # PREFILL while the prompt isn't fully processed; DECODE after.
            if n_prompt is not None and n_computed is not None:
                state = "PREFILL" if n_computed < n_prompt else "DECODE"
            else:
                state = "UNKNOWN"

            # Instantaneous decode rate from the delta since this req's last
            # snapshot. O(1) dict lookup per slot.
            inst_rate = None
            prev = _S.rate_track.get(req_id)
            if prev is not None:
                prev_len, prev_ts = prev
                dt = now - prev_ts
                if dt > 0 and out_len >= prev_len:
                    inst_rate = (out_len - prev_len) / dt
                    agg_rate += inst_rate
            _S.rate_track[req_id] = (out_len, now)

            slots.append(
                {
                    "slot_idx": i,
                    "req_id": req_id,
                    "state": state,
                    "num_prompt_tokens": n_prompt,
                    "num_computed_tokens": n_computed,
                    "out_len": out_len,
                    "inst_rate": round(inst_rate, 2) if inst_rate is not None else None,
                }
            )

        # Drop rate-tracking entries for requests no longer in the batch.
        for stale in [k for k in _S.rate_track if k not in live_ids]:
            # Keep entries that were just admitted (seeded with out_len 0 and a
            # recent ts) so their first real snapshot still has a baseline.
            _, ts = _S.rate_track[stale]
            if (now - ts) > 5.0:
                _S.rate_track.pop(stale, None)

        num_waiting = None
        if scheduler_output is not None:
            # Best-effort: model_runner sees scheduler_output, not the full
            # scheduler queue. Surface a waiting count only if it's there.
            num_waiting = getattr(scheduler_output, "num_waiting", None)

        snap = {
            "schema": SCHEMA_VERSION,
            "event": EVENT_STEP_SNAPSHOT,
            "ts": now,
            "step_idx": _S.step_idx,
            "num_running": len(slots),
            "num_waiting": num_waiting,
            "agg_rate": round(agg_rate, 2),
            "slots": slots,
        }
        _write_snapshot(snap)
        if _S.snapshots_to_jsonl and _S.write_events:
            _append_event(snap)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Event 3: request_completed
# --------------------------------------------------------------------------- #
def emit_request_completed(input_batch: Any, req_id: str) -> None:
    """Hook: InputBatch.remove_request(), called at the TOP before the slot's
    state is cleared (req_output_token_ids[idx] is set to None in-method).

    O(1) per request. Note: input_batch has no finish_reason, so we report
    what's available (out_len, ISL) and let the viewer derive ttft/mean-rate
    from the admit event + snapshots it already saw.
    """
    if not enabled():
        return
    try:
        idx = None
        idx_map = getattr(input_batch, "req_id_to_index", None)
        if idx_map is not None:
            idx = idx_map.get(req_id)

        out_len = None
        n_prompt = None
        if idx is not None:
            out_token_ids = getattr(input_batch, "req_output_token_ids", None)
            if out_token_ids is not None and idx < len(out_token_ids):
                ot = out_token_ids[idx]
                out_len = len(ot) if ot is not None else None
            num_prompt = getattr(input_batch, "num_prompt_tokens", None)
            if num_prompt is not None:
                n_prompt = int(num_prompt[idx])

        _S.rate_track.pop(req_id, None)
        evt = {
            "schema": SCHEMA_VERSION,
            "event": EVENT_REQUEST_COMPLETED,
            "ts": time.time(),
            "req_id": req_id,
            "slot_idx": idx,
            "isl": n_prompt,
            "out_len": out_len,
        }
        if _S.write_events:
            _append_event(evt)
    except Exception:
        pass


def _reset_for_test() -> None:
    """Test-only: clear cached config so a new TT_INSTRUMENT env takes effect."""
    global _S
    _S = _State()
