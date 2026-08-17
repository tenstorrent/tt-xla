# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Env-gated, hot-path-safe serving telemetry for the vLLM TT engine.
# Instruments the two layers that answer different questions about a serving run:
#
#   * SchedulerTelemetry  (AscendScheduler, EngineCore process) -- INTENT: what
#     each step decided and why work stalled (prefill blocking decode,
#     preemption, queue depth, KV/batch utilization).
#   * RunnerTelemetry     (TTModelRunner[V2], Worker process)   -- REALITY: batch
#     occupancy, executed batch composition, prefill/decode pass split, and the
#     actual decode rate.
#
# The two run in separate processes and cannot share a Python object, so each
# writes its own JSON-lines sink; records carry ``request_id`` and a monotonic
# step index so a consumer joins them offline.
#
# Hard rules:
#   - STDLIB ONLY. This lives in the EngineCore / Worker hot path; never pull in
#     anything heavy.
#   - Disabled path is a single cached ``enabled`` bool check: no allocation,
#     no I/O, must never perturb the inference it measures.
#   - NO per-step disk I/O at the default flush interval -- per-step writes would
#     distort the decode rate being measured. Records accumulate in memory and
#     flush on a wall-clock interval, at request completion, and at shutdown.
#     TTXLA_TELEMETRY_FLUSH_MS=0 means "no minimum gap", i.e. flush every step:
#     an explicit opt-in for a live viewer or a test that wants records
#     immediately, at the cost of the guarantee above. Do not use it to measure
#     decode rate.
#   - Per-step work is O(active_slots), never O(output_len).
#
# Gating is resolved in ``TTPlatform.check_and_update_config`` (env overrides
# ``additional_config``, mirroring ``prefill_kv_watermark``) and passed to the
# collector constructors; ``resolve_config`` below re-reads the env so a
# collector built directly in a test still honors ``TTXLA_TELEMETRY``.
#
# Env vars (each overrides the matching ``additional_config`` knob):
#   TTXLA_TELEMETRY           truthy ("1","true","yes","on") enables emission
#   TTXLA_TELEMETRY_DIR       sink directory (default: ./tt_telemetry)
#   TTXLA_TELEMETRY_FLUSH_MS  min gap between disk flushes (default: 1000)
#
# Sinks (all under the configured dir):
#   scheduler.jsonl       append-only per-step scheduler-decision records
#   runner.jsonl          append-only per-step runner records + per-request
#                         completion records
#   runner_snapshot.json  latest slot-occupancy snapshot, atomically
#                         overwritten, for a cheap "current state" view

import atexit
import json
import os
import time
from typing import Any, Optional

# Bump when the record field layout changes so consumers can detect mismatches.
SCHEMA_VERSION = 1

SCHEDULER_FILENAME = "scheduler.jsonl"
RUNNER_FILENAME = "runner.jsonl"
RUNNER_SNAPSHOT_FILENAME = "runner_snapshot.json"

DEFAULT_DIR = "./tt_telemetry"
DEFAULT_FLUSH_MS = 1000.0

_TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in _TRUTHY


def resolve_config(
    enabled: Optional[bool] = None,
    directory: Optional[str] = None,
    flush_ms: Optional[float] = None,
) -> tuple[bool, str, float]:
    """Resolve telemetry settings, letting env vars override the passed config.

    ``TTPlatform.check_and_update_config`` already folds the env overrides into
    ``additional_config``; re-reading here keeps a collector constructed
    directly (e.g. in a unit test) consistent with the env.

    Returns ``(enabled, directory, flush_seconds)``.
    """
    env_enabled = os.environ.get("TTXLA_TELEMETRY")
    if env_enabled is not None:
        enabled = _is_truthy(env_enabled)
    enabled = bool(enabled)

    directory = (
        os.environ.get("TTXLA_TELEMETRY_DIR", "").strip()
        or (directory or "").strip()
        or DEFAULT_DIR
    )

    raw_flush = os.environ.get("TTXLA_TELEMETRY_FLUSH_MS")
    if raw_flush is None:
        raw_flush = flush_ms
    try:
        flush_ms_val = float(raw_flush) if raw_flush is not None else DEFAULT_FLUSH_MS
    except (TypeError, ValueError):
        flush_ms_val = DEFAULT_FLUSH_MS
    return enabled, directory, max(0.0, flush_ms_val / 1000.0)


def reset_sinks(directory):
    """Delete stale sink files so a fresh run starts empty.

    Meant to run ONCE at engine startup, before the per-process collectors are
    constructed, so a live viewer never shows a prior run's data during the new
    run's (long) warmup. Each collector truncates its own JSON-lines file on
    construction, but that is not enough on its own: the scheduler collector is
    built only after warmup, and ``runner_snapshot.json`` is never truncated on
    init (only overwritten on the first step) -- so both would otherwise linger
    through the warmup window. Best-effort; missing files/dir are ignored.
    """
    for name in (SCHEDULER_FILENAME, RUNNER_FILENAME, RUNNER_SNAPSHOT_FILENAME):
        try:
            os.remove(os.path.join(directory, name))
        except OSError:
            pass


class _JsonlSink:
    """Base collector: buffered JSON-lines writing with interval flush.

    Records are appended to an in-memory buffer and written to disk only on
    ``flush`` (interval / completion / shutdown), so the measured decode loop is
    never blocked on per-step I/O. When disabled, ``enabled`` is False and every
    hook is a guarded no-op.
    """

    def __init__(
        self,
        enabled: bool,
        directory: str,
        flush_s: float,
        jsonl_name: str,
        snapshot_name: Optional[str] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._flush_s = flush_s
        self._jsonl_path: Optional[str] = None
        self._snapshot_path: Optional[str] = None
        self._buf: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self.step_idx = 0
        if not self.enabled:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            self._jsonl_path = os.path.join(directory, jsonl_name)
            if snapshot_name is not None:
                self._snapshot_path = os.path.join(directory, snapshot_name)
            # Truncate any stale sink so a consumer doesn't replay a prior run.
            open(self._jsonl_path, "w").close()
            # Flush buffered records at shutdown so the tail isn't lost.
            atexit.register(self.flush)
        except OSError:
            # Never crash the engine over telemetry; just disable.
            self.enabled = False

    def _emit(self, record: dict[str, Any]) -> None:
        """Buffer one record in memory (no I/O)."""
        self._buf.append(record)

    def _maybe_flush(self, now: float) -> None:
        # flush_s == 0 flushes every step: the documented meaning of
        # FLUSH_MS=0, relied on by the tests and the live dashboard.
        if (now - self._last_flush) >= self._flush_s:
            self.flush(now)

    def flush(self, now: Optional[float] = None) -> None:
        """Append buffered records to the JSON-lines sink. Best-effort."""
        if now is None:
            now = time.monotonic()
        self._last_flush = now
        if not self.enabled or not self._buf or self._jsonl_path is None:
            return
        try:
            with open(self._jsonl_path, "a") as f:
                for record in self._buf:
                    f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except (OSError, TypeError, ValueError):
            pass
        self._buf.clear()

    def _write_snapshot(self, obj: dict[str, Any]) -> None:
        """Atomically overwrite the snapshot file (temp + os.replace) so a
        reader never sees a half-written file."""
        if self._snapshot_path is None:
            return
        try:
            tmp = self._snapshot_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            os.replace(tmp, self._snapshot_path)
        except (OSError, TypeError, ValueError):
            pass


class SchedulerTelemetry(_JsonlSink):
    """Per-step scheduler-decision telemetry for ``AscendScheduler``.

    One record per ``schedule()`` call capturing the decision (prefill vs decode
    split, preemption, queue depth) and utilization (KV pool, running batch).
    Cumulative counters let a consumer read totals without re-aggregating.
    """

    def __init__(self, enabled: bool, directory: str, flush_s: float) -> None:
        super().__init__(enabled, directory, flush_s, SCHEDULER_FILENAME)
        self._total_preempted = 0
        self._total_decode_stalled_steps = 0
        self._total_watermark_rejects = 0
        self._total_b1_cap_hits = 0

    def on_schedule(
        self,
        *,
        num_running: int,
        max_running: int,
        num_waiting: int,
        num_free_blocks: int,
        num_total_blocks: int,
        prefill_new: int,
        prefill_resumed: int,
        prefill_partial: int,
        running_scheduled: int,
        preempted: int,
        decode_gated: bool,
        decodes_displaced: int,
        total_scheduled_tokens: int,
        watermark_rejects: int = 0,
        b1_cap_hit: bool = False,
    ) -> None:
        """Hook: end of ``AscendScheduler.schedule()`` (append-only tail)."""
        if not self.enabled:
            return
        try:
            now = time.monotonic()
            self.step_idx += 1
            self._total_preempted += preempted
            if decode_gated:
                self._total_decode_stalled_steps += 1
            self._total_watermark_rejects += watermark_rejects
            if b1_cap_hit:
                self._total_b1_cap_hits += 1

            kv_util = (
                (num_total_blocks - num_free_blocks) / num_total_blocks
                if num_total_blocks > 0
                else None
            )
            batch_util = num_running / max_running if max_running > 0 else None

            self._emit(
                {
                    "schema": SCHEMA_VERSION,
                    "layer": "scheduler",
                    "ts": time.time(),
                    "step": self.step_idx,
                    "num_running": num_running,
                    "max_running": max_running,
                    "num_waiting": num_waiting,
                    "batch_util": (
                        round(batch_util, 4) if batch_util is not None else None
                    ),
                    "kv_util": round(kv_util, 4) if kv_util is not None else None,
                    "num_free_blocks": num_free_blocks,
                    "num_total_blocks": num_total_blocks,
                    "prefill_new": prefill_new,
                    "prefill_resumed": prefill_resumed,
                    "prefill_partial": prefill_partial,
                    "running_scheduled": running_scheduled,
                    "preempted": preempted,
                    # Prefill-first gate; decodes_displaced counts the running
                    # decodes that had to wait.
                    "decode_gated": bool(decode_gated),
                    "decodes_displaced": decodes_displaced,
                    "watermark_rejects": watermark_rejects,
                    "b1_cap_hit": bool(b1_cap_hit),
                    "total_scheduled_tokens": total_scheduled_tokens,
                    "cum_preempted": self._total_preempted,
                    "cum_decode_stalled_steps": self._total_decode_stalled_steps,
                    "cum_watermark_rejects": self._total_watermark_rejects,
                    "cum_b1_cap_hits": self._total_b1_cap_hits,
                }
            )
            self._maybe_flush(now)
        except Exception:
            # Telemetry must never take down the engine.
            pass


class RunnerTelemetry(_JsonlSink):
    """Slot / decode-rate telemetry for the model runner.

    Reads a slot view of the runner's batch, so per-step work is O(active_slots).
    Emits per-step records (occupancy, prefill/decode pass split, decode rate)
    plus per-request completion records, and keeps a latest-state snapshot for
    live viewing.

    The view is whatever the runner passes to ``on_step``: the v2 runner hands
    over its persistent slot table (``TTRequestState``) directly; the v1 runner
    builds ``model_runner._V1SlotView`` over its condensing ``InputBatch``. One
    consequence is worth knowing when reading the output: v1 row indices are not
    stable identities -- ``InputBatch.condense`` compacts rows when a request
    leaves -- so a v1 slot timeline shows row reuse where v2 shows residency.
    """

    def __init__(self, enabled: bool, directory: str, flush_s: float) -> None:
        super().__init__(
            enabled,
            directory,
            flush_s,
            RUNNER_FILENAME,
            snapshot_name=RUNNER_SNAPSHOT_FILENAME,
        )
        self._last_step_ts: Optional[float] = None

    def on_request_admitted(
        self,
        req_id: str,
        slot: int,
        prompt_len: int,
        prefill_len: int,
        *,
        num_cached_tokens: int = 0,
        readmission: bool = False,
    ) -> None:
        """Hook: the runner's admission path (per admitted row). O(1).

        ``prefill_len`` is the absolute prefill boundary -- the full known prefix
        before generation starts -- not the number of tokens left to compute.

        The two runners signal a prefix-cache hit differently: v2 carries a
        ``prefill_token_ids`` prefix that can exceed the prompt, so
        ``prefill_len > prompt_len``; v1 has no such field and instead admits the
        request with ``num_computed_tokens`` already advanced. Accept both so the
        record means the same thing on either runner.

        ``readmission`` distinguishes a request entering the batch again from a
        first admission, so consumers counting requests served -- or hunting
        genuine cross-request prefix hits -- can filter it out. The runners reach
        this state differently:

        * v1 removes any request it did not schedule this step from the batch and
          re-adds it later, usually at a different row. This is common, so
          admission records routinely outnumber requests, and a re-admitted
          request reports ``prefix_cache_hit`` against its own prefix.
        * v2 keeps a request in its slot across steps, so this only fires when an
          id arrives while a stale slot still holds it (abort + resubmit). A v2
          request resumed from preemption instead arrives as a fresh request and
          shows up as nonzero ``num_cached_tokens`` / ``prefill_len >
          prompt_len``, not as a re-admission.
        """
        if not self.enabled:
            return
        try:
            self._emit(
                {
                    "schema": SCHEMA_VERSION,
                    "layer": "runner",
                    "event": "request_admitted",
                    "ts": time.time(),
                    "step": self.step_idx,
                    "request_id": req_id,
                    "slot": slot,
                    "prompt_len": prompt_len,
                    "prefill_len": prefill_len,
                    "num_cached_tokens": num_cached_tokens,
                    "prefix_cache_hit": prefill_len > prompt_len
                    or num_cached_tokens > 0,
                    "readmission": readmission,
                }
            )
        except Exception:
            pass

    def on_request_completed(
        self, req_id: str, slot: int, prompt_len: int, output_len: int
    ) -> None:
        """Hook: the runner's request-teardown path (per freed slot). O(1).

        Flushes so a completed request is visible promptly to a consumer.
        """
        if not self.enabled:
            return
        try:
            self._emit(
                {
                    "schema": SCHEMA_VERSION,
                    "layer": "runner",
                    "event": "request_completed",
                    "ts": time.time(),
                    "step": self.step_idx,
                    "request_id": req_id,
                    "slot": slot,
                    "prompt_len": prompt_len,
                    "output_len": output_len,
                }
            )
            self.flush()
        except Exception:
            pass

    def on_step(
        self,
        req_states: Any,
        *,
        prefill_passes: int,
        decode_passes: int,
        emitted_tokens: int,
    ) -> None:
        """Hook: end of the runner's ``sample_tokens``. O(active_slots).

        ``emitted_tokens`` counts *tokens* emitted this step, not emitting rows
        (rows still prefilling emit none). Decode rate is emitted-tokens /
        wall-time since the previous step, i.e. *accepted* tokens/step. On the v1
        runner that distinction is live rather than theoretical: with spec decode
        a row can accept several tokens in one step, and the caller sums
        the per-row accepted lengths (``gen_lens``) rather than counting rows.
        """
        if not self.enabled:
            return
        try:
            now = time.monotonic()
            self.step_idx += 1

            occupied = len(req_states.req_id_to_index)
            free = len(req_states.free_indices)
            total_slots = occupied + free

            prefilling = 0
            decoding = 0
            slots: list[dict[str, Any]] = []
            for req_id, slot in req_states.req_id_to_index.items():
                computed = int(req_states.num_computed_tokens[slot])
                prefill_len = int(req_states.prefill_len[slot])
                state = "PREFILL" if computed < prefill_len else "DECODE"
                if state == "PREFILL":
                    prefilling += 1
                else:
                    decoding += 1
                slots.append(
                    {
                        "slot": slot,
                        "request_id": req_id,
                        "state": state,
                        "num_computed_tokens": computed,
                        "prefill_len": prefill_len,
                        "total_len": int(req_states.total_len[slot]),
                    }
                )

            dt = None if self._last_step_ts is None else (now - self._last_step_ts)
            self._last_step_ts = now
            decode_rate = (emitted_tokens / dt) if (dt is not None and dt > 0) else None

            record = {
                "schema": SCHEMA_VERSION,
                "layer": "runner",
                "event": "step",
                "ts": time.time(),
                "step": self.step_idx,
                "slots_occupied": occupied,
                "slots_free": free,
                "slots_total": total_slots,
                "slot_util": (
                    round(occupied / total_slots, 4) if total_slots > 0 else None
                ),
                "num_prefilling": prefilling,
                "num_decoding": decoding,
                # Per-step pass split: a pass is a prefill pass if any row in it
                # is still below its prefill_len, else a decode pass.
                "prefill_passes": prefill_passes,
                "decode_passes": decode_passes,
                "emitted_tokens": emitted_tokens,
                "step_dt_s": round(dt, 6) if dt is not None else None,
                "decode_rate_toks_per_s": (
                    round(decode_rate, 2) if decode_rate is not None else None
                ),
            }
            self._emit(record)
            self._write_snapshot({**record, "slots": slots})
            self._maybe_flush(now)
        except Exception:
            pass
