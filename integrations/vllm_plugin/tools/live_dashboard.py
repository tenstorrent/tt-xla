#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Live inference dashboard for the TT vLLM serving path.
#
# A keyboard-driven TUI that shows, per request/slot, what the server is doing
# in real time: prefill vs decode, per-stream tok/s, TTFT, and how firing new
# requests perturbs in-flight streams (the decode "hitch" during a burst's
# prefill -- the headline visualization).
#
# Three data sources (the same renderer + model serve all three):
#   --source demo      synthetic data, NO server. Run this first to see the UI.
#   --source client    fire OpenAI-compatible requests at a running server and
#                       INFER state from SSE token cadence (Approach A: zero
#                       server changes; interference is inferred, not read).
#   --source snapshot  read the engine's own telemetry (Approach B): true slots,
#                       real PREFILL vs DECODE, num_waiting. Requires the server
#                       to run with TT_INSTRUMENT=1 (see vllm_tt/instrumentation.py).
#
# Two renderers:
#   textual (default in a tty) -- a scrollable, selectable DataTable + live
#            footer. Select a row to target it with `k`. Needs `textual`
#            (pip install -r integrations/vllm_plugin/tools/requirements.txt).
#   plain   (--plain, non-tty, or if textual is missing) -- stdlib ANSI, zero
#            extra deps; used for headless/CI runs (with --exit-after).
#
# Keys (in a tty):
#   n  launch 1 request        b  launch a burst (default 4)
#   k  cancel selected row      t  toggle token-text / counter mode
#   p  toggle prefill highlight  q  quit (aborts all streams cleanly)
#   1..9  launch that many at once   (plain renderer cancels the newest)
# (client + demo are interactive; snapshot is read-only except t/p/q.)
#
# Honest limits of --source client (Approach A): it CANNOT distinguish prefill
# from queue-wait (both look like "no token yet" -- labelled TTFT/PREFILL), and
# it has no true slot IDs / KV usage / explicit preemption. Interference is
# inferred from rate, not read from the server. Use --source snapshot for ground
# truth once the server runs instrumented.

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ----------------------------------------------------------------------------
# Unified per-stream model -- schema-compatible with instrumentation.py's
# step_snapshot "slots", so the snapshot source can populate it directly and a
# client/demo source can fill the same shape. One model, three sources.
# ----------------------------------------------------------------------------
S_CONNECTING = "CONNECTING"
S_PREFILL = "PREFILL"  # request sent / scheduled, no output token yet (TTFT window)
S_DECODE = "DECODE"
S_STALLED = "STALLED"  # decode paused while another request prefills (demo only)
S_DONE = "DONE"
S_CANCELLED = "CANCELLED"
S_ERROR = "ERROR"

_ACTIVE_STATES = {S_CONNECTING, S_PREFILL, S_DECODE, S_STALLED}


@dataclass
class StreamState:
    id: str
    state: str = S_CONNECTING
    # Timing
    t_start: float = field(default_factory=time.time)
    t_first: Optional[float] = None  # first token arrival
    t_last: Optional[float] = None
    ttft: Optional[float] = None
    # Counts
    isl: Optional[int] = None
    n_tokens: int = 0
    osl_target: Optional[int] = None
    # True-slot fields (snapshot source); None for client/demo.
    slot_idx: Optional[int] = None
    num_prompt: Optional[int] = None
    num_computed: Optional[int] = None
    # Derived
    rate: float = 0.0  # EWMA tok/s
    finish_reason: Optional[str] = None
    text_tail: str = ""

    def note_token(self, now: float, text: str = "") -> None:
        if self.t_first is None:
            self.t_first = now
            self.ttft = now - self.t_start
            self.state = S_DECODE
        elif self.t_last is not None:
            dt = now - self.t_last
            # Ignore sub-ms gaps (batched SSE chunks) so 1/dt can't explode.
            if dt > 1e-3:
                inst = 1.0 / dt
                # EWMA so the rate reacts to hitches but isn't jumpy.
                self.rate = inst if self.rate == 0 else 0.6 * self.rate + 0.4 * inst
        self.t_last = now
        self.n_tokens += 1
        if text:
            self.text_tail = (self.text_tail + text)[-120:]

    @property
    def elapsed(self) -> float:
        end = (
            self.t_last
            if self.state not in _ACTIVE_STATES and self.t_last
            else time.time()
        )
        return end - self.t_start

    @property
    def is_active(self) -> bool:
        return self.state in _ACTIVE_STATES


class Model:
    """Shared in-memory state the renderer reads and sources write."""

    def __init__(self):
        self.streams: dict[str, StreamState] = {}
        self.order: list[str] = []  # insertion order for stable display
        self.counter_mode: bool = False  # token text vs counter
        self.highlight_prefill: bool = True
        self.source_label: str = ""
        self.num_waiting: Optional[int] = None
        self.notice: str = ""
        self._seq = 0

    def new_id(self, prefix: str = "s") -> str:
        self._seq += 1
        return f"{prefix}{self._seq}"

    def add(self, st: StreamState) -> None:
        self.streams[st.id] = st
        self.order.append(st.id)

    def upsert_slot(self, slot: dict, now: float) -> None:
        """Populate/refresh a stream from a snapshot slot dict."""
        rid = str(slot.get("req_id"))
        st = self.streams.get(rid)
        if st is None:
            st = StreamState(id=rid)
            self.add(st)
        st.slot_idx = slot.get("slot_idx")
        st.num_prompt = slot.get("num_prompt_tokens")
        st.num_computed = slot.get("num_computed_tokens")
        st.isl = st.num_prompt
        st.n_tokens = slot.get("out_len", st.n_tokens) or st.n_tokens
        snap_state = slot.get("state")
        if snap_state in (S_PREFILL, S_DECODE):
            st.state = snap_state
            if snap_state == S_DECODE and st.t_first is None:
                st.t_first = now
                st.ttft = max(0.0, now - st.t_start)
        r = slot.get("inst_rate")
        if r is not None:
            st.rate = r
        st.t_last = now

    def active(self) -> list[StreamState]:
        return [self.streams[i] for i in self.order if self.streams[i].is_active]

    def visible(self, limit: int = 24) -> list[StreamState]:
        # Show active first, then most-recently-finished; cap for the viewport.
        act = [self.streams[i] for i in self.order if self.streams[i].is_active]
        done = [self.streams[i] for i in self.order if not self.streams[i].is_active]
        return (act + list(reversed(done)))[:limit]

    def agg_rate(self) -> float:
        return sum(s.rate for s in self.active())

    def total_tokens(self) -> int:
        return sum(s.n_tokens for s in self.streams.values())

    def prune(self, keep_done: int = 12) -> None:
        done = [i for i in self.order if not self.streams[i].is_active]
        for i in done[:-keep_done] if len(done) > keep_done else []:
            self.streams.pop(i, None)
            self.order.remove(i)


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------
class Source:
    """Common interface so renderer/keys don't care which source is active."""

    interactive = False

    async def run(self):  # background producer loop
        ...

    async def launch(self, n: int = 1): ...

    async def cancel(self, target: Optional[str] = None): ...

    async def aclose(self): ...


class DemoSource(Source):
    """Synthesizes arrivals, prefill windows, decode, and the interference a new
    request's prefill imposes on in-flight decode. No server.

    interference shapes (--demo-interference):
      freeze    in-flight decode emits NO tokens while any request prefills
                (state -> STALLED, OUT frozen), then resumes with a one-shot
                latency-spike blip. This mirrors TT, where prefill and decode
                are typically separate steps so a prefill step doesn't advance
                decode -- a pause, not a slow-down.
      slowdown  in-flight decode continues at a reduced rate during prefill.
                A gentler, chunked-prefill-like approximation.

    Both magnitudes are invented for teaching; trust --source snapshot/client
    for what the real chip does."""

    interactive = True

    def __init__(
        self,
        model: Model,
        base_rate=22.0,
        isl=512,
        osl=160,
        auto=True,
        interference="freeze",
    ):
        self.m = model
        self.base_rate = base_rate
        self.isl = isl
        self.osl = osl
        self.auto = auto
        self.interference = interference
        self._accum: dict[str, float] = {}  # fractional token accumulator
        self._prefill_until: dict[str, float] = {}
        self._stalled_since: dict[str, float] = {}  # freeze: when decode paused
        self._spike_ticks: dict[str, int] = {}  # freeze: post-resume blip
        self._stop = False
        self.m.source_label = f"demo (synthetic, {interference})"

    def _admit(self):
        st = StreamState(
            id=self.m.new_id("d"),
            state=S_PREFILL,
            isl=self.isl,
            osl_target=self.osl,
            num_prompt=self.isl,
            num_computed=0,
        )
        self.m.add(st)
        self._accum[st.id] = 0.0
        # Prefill takes ~ proportional to ISL; larger ISL = longer TTFT window.
        self._prefill_until[st.id] = time.time() + min(2.2, 0.4 + self.isl / 700.0)

    async def launch(self, n: int = 1):
        for _ in range(n):
            self._admit()
        self.m.notice = f"launched {n} synthetic request(s)"

    async def cancel(self, target: Optional[str] = None):
        act = self.m.active()
        if not act:
            return
        st = self.m.streams.get(target) if target else act[-1]
        if st:
            st.state = S_CANCELLED
            st.finish_reason = "cancelled"
            self.m.notice = f"cancelled {st.id}"

    def _decode_rate(self, st: StreamState, prefilling: bool, now: float) -> float:
        """Per-tick decode rate for one in-flight stream given whether some
        other request is prefilling. Also drives the STALLED state + the
        one-shot resume latency spike in freeze mode."""
        if self.interference == "slowdown":
            st.state = S_DECODE
            return self.base_rate * (0.3 if prefilling else 1.0)

        # freeze: decode is paused entirely while anything prefills.
        if prefilling:
            if st.state != S_STALLED:
                st.state = S_STALLED
                self._stalled_since[st.id] = now
            return 0.0

        if st.state == S_STALLED:
            # Just resumed: model the long inter-token gap as a brief low-rate
            # blip (the latency spike) before recovering to full speed.
            stall = max(1e-3, now - self._stalled_since.pop(st.id, now))
            st.state = S_DECODE
            self._spike_ticks[st.id] = 3  # ~0.3s of post-stall recovery
            return 1.0 / stall

        if self._spike_ticks.get(st.id, 0) > 0:
            self._spike_ticks[st.id] -= 1
            return self.base_rate * 0.15
        return self.base_rate

    async def run(self):
        last_auto = time.time()
        while not self._stop:
            now = time.time()
            if self.auto and (now - last_auto) > 2.6 and len(self.m.active()) < 6:
                self._admit()
                last_auto = now
            # A new request's prefill interferes with everyone's decode.
            prefilling = any(
                s.state == S_PREFILL and self._prefill_until.get(s.id, 0) > now
                for s in self.m.active()
            )
            for st in list(self.m.active()):
                if st.state == S_PREFILL:
                    if self._prefill_until.get(st.id, 0) <= now:
                        # First token: enter DECODE and record TTFT directly
                        # (note_token's 1/dt rate model is for real streams).
                        st.t_first = now
                        st.ttft = now - st.t_start
                        st.state = S_DECODE
                        st.t_last = now
                        self._accum[st.id] = 0.0
                    continue
                if st.state in (S_DECODE, S_STALLED):
                    cur_rate = self._decode_rate(st, prefilling, now)
                    if cur_rate > 0:
                        self._accum[st.id] += cur_rate * 0.1
                        n_new = int(self._accum[st.id])
                        if n_new:
                            self._accum[st.id] -= n_new
                            st.n_tokens += n_new
                            st.text_tail = (st.text_tail + "x" * n_new)[-120:]
                            st.t_last = now
                    st.rate = cur_rate
                    if st.osl_target and st.n_tokens >= st.osl_target:
                        st.state = S_DONE
                        st.finish_reason = "length"
            self.m.num_waiting = max(0, len(self.m.active()) - 4)
            self.m.prune()
            await asyncio.sleep(0.1)

    async def aclose(self):
        self._stop = True


class SnapshotSource(Source):
    """Reads the engine's telemetry (Approach B): polls snapshot.json for the
    current step and tails events.jsonl for admit/complete. Read-only truth."""

    interactive = False

    def __init__(self, model: Model, directory: str, poll_hz=10.0):
        self.m = model
        self.dir = directory
        self.snap_path = os.path.join(directory, "snapshot.json")
        self.events_path = os.path.join(directory, "events.jsonl")
        self.interval = 1.0 / poll_hz
        self._stop = False
        self._events_pos = 0
        self.m.source_label = f"snapshot ({directory})"

    async def run(self):
        while not self._stop:
            # Read events first (may create streams), then stamp `now` so a
            # just-created stream's t_start can't post-date the snapshot clock.
            self._read_events()
            self._read_snapshot(time.time())
            self.m.prune()
            await asyncio.sleep(self.interval)

    def _read_snapshot(self, now: float):
        try:
            with open(self.snap_path) as f:
                snap = json.load(f)
        except (OSError, ValueError):
            return
        if snap.get("event") != "step_snapshot":
            return
        self.m.num_waiting = snap.get("num_waiting")
        live = set()
        for slot in snap.get("slots", []):
            self.m.upsert_slot(slot, now)
            live.add(str(slot.get("req_id")))
        # Anything previously active but absent from the latest step is done.
        for st in self.m.active():
            if st.id not in live and st.slot_idx is not None:
                st.state = S_DONE

    def _read_events(self):
        try:
            with open(self.events_path) as f:
                f.seek(self._events_pos)
                lines = f.readlines()
                self._events_pos = f.tell()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except ValueError:
                continue
            kind = evt.get("event")
            rid = str(evt.get("req_id"))
            if kind == "request_admitted":
                if rid not in self.m.streams:
                    st = StreamState(
                        id=rid,
                        state=S_PREFILL,
                        isl=evt.get("isl"),
                        slot_idx=evt.get("slot_idx"),
                    )
                    self.m.add(st)
                    self.m.notice = f"admitted {rid} (ISL {evt.get('isl')})"
            elif kind == "request_completed":
                st = self.m.streams.get(rid)
                if st:
                    st.state = S_DONE
                    st.finish_reason = "completed"
                    if evt.get("out_len") is not None:
                        st.n_tokens = evt["out_len"]

    async def aclose(self):
        self._stop = True


class ClientSource(Source):
    """Approach A: fire OpenAI-compatible streaming requests and infer state
    from SSE timing. One asyncio task per stream. Distinct nonce per prompt to
    defeat prefix caching (else TTFT is fake on repeats)."""

    interactive = True

    def __init__(self, model: Model, cfg: dict):
        self.m = model
        self.cfg = cfg
        self._tasks: dict[str, asyncio.Task] = {}
        self._session = None
        self.m.source_label = f"client {cfg['host']}:{cfg['port']} [{cfg['model']}]"

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    def _build_body(self, nonce: str) -> tuple[str, dict]:
        c = self.cfg
        prompt = f"[{nonce}] {c['prompt']}"
        body = {
            "model": c["model"],
            "max_tokens": c["max_tokens"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if c["temperature"] is not None:
            body["temperature"] = c["temperature"]
        if c["top_p"] is not None:
            body["top_p"] = c["top_p"]
        if c["top_k"] is not None:
            body["top_k"] = c["top_k"]
        if c["repetition_penalty"] is not None:
            body["repetition_penalty"] = c["repetition_penalty"]
        if c["seed"] is not None:
            body["seed"] = c["seed"]
        if c["ignore_eos"]:
            body["ignore_eos"] = True
        if c["completions"]:
            endpoint = "completions"
            body["prompt"] = prompt
        else:
            endpoint = "chat/completions"
            body["messages"] = [{"role": "user", "content": prompt}]
        return endpoint, body

    async def _stream(self, st: StreamState):
        c = self.cfg
        nonce = uuid.uuid4().hex[:8]
        endpoint, body = self._build_body(nonce)
        url = f"http://{c['host']}:{c['port']}/v1/{endpoint}"
        headers = {"Content-Type": "application/json"}
        if c.get("api_key"):
            headers["Authorization"] = f"Bearer {c['api_key']}"
        session = await self._ensure_session()
        st.state = S_PREFILL
        try:
            async with session.post(url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                buf = ""
                async for chunk in resp.content.iter_any():
                    buf += chunk.decode("utf-8", "replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: ") :]
                        if data == "[DONE]":
                            st.state = S_DONE
                            return
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue
                        usage = obj.get("usage")
                        if usage:
                            st.isl = usage.get("prompt_tokens", st.isl)
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        ch = choices[0]
                        text = (
                            ch.get("text") or ch.get("delta", {}).get("content") or ""
                        )
                        if text:
                            st.note_token(time.time(), text)
                        if ch.get("finish_reason"):
                            st.finish_reason = ch["finish_reason"]
                            st.state = S_DONE
                            return
            if st.state != S_DONE:
                st.state = S_DONE
        except asyncio.CancelledError:
            st.state = S_CANCELLED
            st.finish_reason = "cancelled"
            raise
        except Exception as e:  # connection/HTTP errors -> show, don't crash
            st.state = S_ERROR
            st.finish_reason = f"{type(e).__name__}"
            self.m.notice = f"{st.id}: {type(e).__name__}: {e}"

    async def launch(self, n: int = 1):
        for _ in range(n):
            st = StreamState(id=self.m.new_id("c"), osl_target=self.cfg["max_tokens"])
            self.m.add(st)
            self._tasks[st.id] = asyncio.create_task(self._stream(st))
        self.m.notice = f"launched {n} request(s)"

    async def cancel(self, target: Optional[str] = None):
        act = self.m.active()
        st = self.m.streams.get(target) if target else (act[-1] if act else None)
        if st and st.id in self._tasks:
            self._tasks[st.id].cancel()
            self.m.notice = f"cancelled {st.id}"

    async def run(self):
        # Reaper: keep finished tasks from accumulating; prune old rows.
        while True:
            for rid, t in list(self._tasks.items()):
                if t.done():
                    self._tasks.pop(rid, None)
            self.m.prune()
            await asyncio.sleep(0.2)

    async def aclose(self):
        for t in self._tasks.values():
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        if self._session is not None:
            await self._session.close()


# ----------------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------------
def _fmt(x, nd=1, dash="-"):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x is not None else dash


def _state_color(state: str) -> str:
    return {
        S_PREFILL: "yellow",
        S_DECODE: "green",
        S_STALLED: "bold red",
        S_DONE: "dim",
        S_CANCELLED: "red",
        S_ERROR: "red",
        S_CONNECTING: "cyan",
    }.get(state, "white")


_TABLE_COLS = [
    "ID",
    "SLOT",
    "STATE",
    "ISL",
    "OUT",
    "tok/s",
    "TTFT",
    "elapsed",
    "text/tokens",
]


def make_textual_app(
    model: Model,
    source: Source,
    burst_n: int,
    start_n: int = 0,
    exit_after: float = 0.0,
):
    """Build the Textual dashboard app. Imports textual lazily and defines the
    App subclass here so the module still imports (for --plain / headless) when
    textual isn't installed. Raises ImportError if textual is missing."""
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import DataTable, Footer, Header, Static

    def cells_for(st: StreamState, m: Model):
        if m.highlight_prefill and st.state == S_PREFILL:
            state = Text(st.state, style="black on yellow")
        else:
            state = Text(st.state, style=_state_color(st.state))
        last = (
            ("#" * min(48, st.n_tokens // 4)) if m.counter_mode else st.text_tail[-60:]
        )
        return [
            st.id,
            str(st.slot_idx) if st.slot_idx is not None else "-",
            state,
            str(st.isl) if st.isl is not None else "-",
            str(st.n_tokens),
            _fmt(st.rate, 1),
            f"{st.ttft:.2f}s" if st.ttft else "-",
            f"{st.elapsed:.1f}s",
            last,
        ]

    class Dashboard(App):
        TITLE = "TT vLLM live dashboard"
        CSS = """
        DataTable { height: 1fr; }
        #stats { height: 3; padding: 0 1; background: $panel; color: $text; }
        """
        BINDINGS = [
            Binding("q", "quit_clean", "quit"),
            Binding("n", "launch_one", "launch"),
            Binding("b", "burst", "burst"),
            Binding("k", "cancel_sel", "cancel"),
            Binding("t", "toggle_counter", "counter"),
            Binding("p", "toggle_prefill", "prefill-hl"),
        ]

        def __init__(self):
            super().__init__()
            self.m = model
            self.source = source
            self._col_keys: list = []
            self._row_keys: set = set()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield DataTable(id="slots", zebra_stripes=True)
            yield Static("", id="stats")
            yield Footer()

        async def on_mount(self) -> None:
            self.sub_title = self.m.source_label
            table = self.query_one(DataTable)
            self._col_keys = list(table.add_columns(*_TABLE_COLS))
            table.cursor_type = "row"
            self.run_worker(self.source.run(), name="source", exclusive=False)
            if start_n:
                await self.source.launch(start_n)
            self.set_interval(0.1, self.refresh_view)
            if exit_after:
                self.set_timer(exit_after, self.action_quit_clean)

        def refresh_view(self) -> None:
            m = self.m
            table = self.query_one(DataTable)
            visible = m.visible()
            vidset = {s.id for s in visible}
            for rid in list(self._row_keys):
                if rid not in vidset:
                    try:
                        table.remove_row(rid)
                    except Exception:
                        pass
                    self._row_keys.discard(rid)
            for st in visible:
                cells = cells_for(st, m)
                if st.id in self._row_keys:
                    for ck, val in zip(self._col_keys, cells):
                        table.update_cell(st.id, ck, val, update_width=False)
                else:
                    table.add_row(*cells, key=st.id)
                    self._row_keys.add(st.id)
            self._update_stats()

        def _update_stats(self) -> None:
            m = self.m
            act = m.active()
            n_pref = sum(1 for s in act if s.state == S_PREFILL)
            n_dec = sum(1 for s in act if s.state == S_DECODE)
            waiting = m.num_waiting if m.num_waiting is not None else "-"
            txt = (
                f"[b]source[/] {m.source_label}    "
                f"[green]decode[/] {n_dec}  [yellow]prefill[/] {n_pref}  "
                f"waiting {waiting}  done {len(m.streams) - len(act)}    "
                f"[b]agg[/] {m.agg_rate():.1f} tok/s   total {m.total_tokens()} tok"
            )
            if m.notice:
                txt += f"\n[cyan]{m.notice}[/]"
            self.query_one("#stats", Static).update(txt)

        async def on_key(self, event) -> None:
            # 1..9 launch that many at once (kept off the footer to reduce noise).
            if event.key.isdigit() and event.key != "0":
                await self._launch(int(event.key))

        async def _launch(self, n: int) -> None:
            if self.source.interactive:
                await self.source.launch(n)
            else:
                self.m.notice = "launch disabled for snapshot source (read-only)"

        async def action_launch_one(self) -> None:
            await self._launch(1)

        async def action_burst(self) -> None:
            await self._launch(burst_n)

        async def action_cancel_sel(self) -> None:
            if not self.source.interactive:
                self.m.notice = "cancel disabled for snapshot source (read-only)"
                return
            target = None
            try:
                table = self.query_one(DataTable)
                key = table.coordinate_to_cell_key(table.cursor_coordinate)
                target = key.row_key.value
            except Exception:
                target = None
            await self.source.cancel(target)

        def action_toggle_counter(self) -> None:
            self.m.counter_mode = not self.m.counter_mode

        def action_toggle_prefill(self) -> None:
            self.m.highlight_prefill = not self.m.highlight_prefill

        async def action_quit_clean(self) -> None:
            try:
                await self.source.aclose()
            finally:
                self.exit()

    return Dashboard()


class PlainRenderer:
    """Stdlib ANSI renderer -- zero extra deps. Same model, simpler look."""

    def __init__(self, model: Model):
        self.m = model

    def _render(self) -> str:
        m = self.m
        out = ["\033[2J\033[H"]  # clear + home
        out.append("== TT vLLM live dashboard ==  source: %s\n" % m.source_label)
        hdr = f"{'ID':<6}{'SLOT':>5} {'STATE':<10}{'ISL':>6}{'OUT':>6}{'tok/s':>8}{'TTFT':>8}{'elaps':>8}  text"
        out.append(hdr + "\n" + "-" * min(len(hdr) + 30, 120) + "\n")
        for st in m.visible():
            mark = "*" if (m.highlight_prefill and st.state == S_PREFILL) else " "
            tail = (
                ("#" * min(30, st.n_tokens // 4))
                if m.counter_mode
                else st.text_tail[-40:]
            )
            out.append(
                f"{mark}{st.id:<5}{(st.slot_idx if st.slot_idx is not None else '-'):>5} "
                f"{st.state:<10}{(st.isl if st.isl is not None else '-'):>6}{st.n_tokens:>6}"
                f"{st.rate:>8.1f}{(self._ttft(st)):>8}{st.elapsed:>7.1f}s  {tail}\n"
            )
        act = m.active()
        n_pref = sum(1 for s in act if s.state == S_PREFILL)
        n_dec = sum(1 for s in act if s.state == S_DECODE)
        waiting = m.num_waiting if m.num_waiting is not None else "-"
        out.append("\n")
        out.append(
            f"decoding {n_dec}  prefill {n_pref}  waiting {waiting}  "
            f"done {len(m.streams) - len(act)}  |  agg {m.agg_rate():.1f} tok/s  "
            f"total {m.total_tokens()} tok\n"
        )
        out.append(
            "keys: n launch  b burst  k cancel  t counter  p prefill-hl  q quit\n"
        )
        if m.notice:
            out.append(">> " + m.notice + "\n")
        return "".join(out)

    @staticmethod
    def _ttft(st):
        return f"{st.ttft:.2f}s" if st.ttft else "-"

    async def loop(self, stop_evt: asyncio.Event):
        try:
            while not stop_evt.is_set():
                sys.stdout.write(self._render())
                sys.stdout.flush()
                await asyncio.sleep(0.15)
        finally:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()


# ----------------------------------------------------------------------------
# Keyboard (raw stdin via add_reader; degrades gracefully if not a tty)
# ----------------------------------------------------------------------------
async def keyboard(source: Source, model: Model, stop_evt: asyncio.Event, burst_n: int):
    if not sys.stdin.isatty():
        model.notice = "stdin not a tty -- keys disabled (running passively)"
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_readable():
        try:
            ch = os.read(fd, 1).decode("utf-8", "replace")
        except OSError:
            return
        queue.put_nowait(ch)

    loop.add_reader(fd, on_readable)
    try:
        while not stop_evt.is_set():
            ch = await queue.get()
            if ch in ("q", "\x03"):  # q or Ctrl-C
                stop_evt.set()
                break
            elif ch == "t":
                model.counter_mode = not model.counter_mode
            elif ch == "p":
                model.highlight_prefill = not model.highlight_prefill
            elif ch == "n" and source.interactive:
                await source.launch(1)
            elif ch == "b" and source.interactive:
                await source.launch(burst_n)
            elif ch == "k" and source.interactive:
                await source.cancel()
            elif ch.isdigit() and ch != "0" and source.interactive:
                await source.launch(int(ch))
    finally:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ----------------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------------
def build_source(args, model: Model) -> Source:
    if args.source == "demo":
        return DemoSource(
            model,
            isl=args.isl,
            osl=args.max_tokens,
            auto=not args.no_auto,
            interference=args.demo_interference,
        )
    if args.source == "snapshot":
        return SnapshotSource(model, args.dir)
    cfg = dict(
        host=args.host,
        port=args.port,
        model=args.model,
        api_key=args.api_key,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        ignore_eos=args.ignore_eos,
        completions=args.completions,
    )
    return ClientSource(model, cfg)


async def amain(args, model: Model, source: Source):
    """Plain (stdlib ANSI) run loop -- used for --plain, non-tty, and headless
    --exit-after runs. The textual renderer is driven separately in main()."""
    stop_evt = asyncio.Event()

    tasks = [
        asyncio.create_task(source.run()),
        asyncio.create_task(PlainRenderer(model).loop(stop_evt)),
        asyncio.create_task(keyboard(source, model, stop_evt, args.burst)),
    ]
    if args.source == "client" and args.start:
        await source.launch(args.start)

    if args.exit_after:

        async def _autostop():
            await asyncio.sleep(args.exit_after)
            stop_evt.set()

        tasks.append(asyncio.create_task(_autostop()))

    await stop_evt.wait()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await source.aclose()


def main():
    p = argparse.ArgumentParser(description="Live TT vLLM inference dashboard")
    p.add_argument("--source", choices=["demo", "client", "snapshot"], default="demo")
    p.add_argument("--plain", action="store_true", help="force stdlib ANSI renderer")
    # client / server
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--api-key", default=os.environ.get("TT_API_KEY", ""))
    p.add_argument(
        "--completions",
        action="store_true",
        help="use /v1/completions instead of /v1/chat/completions",
    )
    p.add_argument(
        "--prompt", default="Write a detailed essay about the history of computing."
    )
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--ignore-eos", action="store_true", help="force full OSL (force_len)"
    )
    p.add_argument(
        "--start", type=int, default=0, help="client: auto-launch N at startup"
    )
    p.add_argument("--burst", type=int, default=4, help="size of a 'b' burst")
    # snapshot
    p.add_argument(
        "--dir",
        default=os.environ.get("TT_INSTRUMENT_DIR", ".tt_instrument"),
        help="snapshot source: TT_INSTRUMENT_DIR to read",
    )
    # demo
    p.add_argument("--isl", type=int, default=512, help="demo: synthetic ISL")
    p.add_argument(
        "--no-auto", action="store_true", help="demo: don't auto-spawn requests"
    )
    p.add_argument(
        "--demo-interference",
        choices=["freeze", "slowdown"],
        default="freeze",
        help="demo: how a prefill perturbs in-flight decode (default: freeze)",
    )
    p.add_argument(
        "--exit-after",
        type=float,
        default=0,
        help="auto-quit after N seconds (headless/CI verification)",
    )
    args = p.parse_args()

    model = Model()
    source = build_source(args, model)

    # Textual in a real terminal; otherwise the stdlib plain renderer (also the
    # path for --plain and headless --exit-after verification).
    want_textual = not args.plain and sys.stdin.isatty() and sys.stdout.isatty()
    if want_textual:
        try:
            app = make_textual_app(
                model,
                source,
                args.burst,
                args.start if args.source == "client" else 0,
                args.exit_after,
            )
        except ImportError:
            print(
                "textual not installed; falling back to --plain. Install with:\n"
                "  pip install -r integrations/vllm_plugin/tools/requirements.txt",
                file=sys.stderr,
            )
            app = None
        if app is not None:
            app.run()
            return

    try:
        asyncio.run(amain(args, model, source))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
