#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Offline analysis of a TT vLLM run from its telemetry (vllm_tt/instrumentation.py).
#
# Reads the append-only events.jsonl produced with TT_INSTRUMENT=1 and prints a
# markdown summary (where time went, overlap/interference, per-request stats).
# With --html it also writes a self-contained, zoomable timeline of the N batch
# slots (no external deps -- inline SVG + a little JS for wheel-zoom / drag-pan).
#
#   python3 analyze_run.py --dir /tmp/tt_instrument/qwen3-8b
#   python3 analyze_run.py --dir <dir> --html run.html --json run.json
#
# Fidelity tiers:
#   events only (request_admitted/_completed)  -> per-request bars + overlap +
#       TTFT/OSL/latency stats (TTFT/mean_rate come from the v2 completion event).
#   + step_snapshots (run server with TT_INSTRUMENT_SNAPSHOTS_JSONL=1)         ->
#       per-slot PREFILL/DECODE/STALLED segments, step-kind breakdown, occupancy,
#       and stall-seconds (interference cost).
#
# Caveat: snapshots are sampled (throttle, default 100ms), so the state timeline
# and time-in-state are a reconstruction, not an exact trace. Lower
# TT_INSTRUMENT_THROTTLE_MS for finer offline runs.

import argparse
import html
import json
import os
import sys

# State classification for the timeline (consumer-side; mirrors the dashboard).
PREFILL, DECODE, STALLED = "prefill", "decode", "stalled"
_COLORS = {
    PREFILL: "#d4a017",
    DECODE: "#2e8b57",
    STALLED: "#c0392b",
    "connecting": "#2980b9",
    "request": "#4a90d2",
}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_events(path):
    admits, completes, snapshots = {}, {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            kind = e.get("event")
            rid = str(e.get("req_id"))
            if kind == "request_admitted":
                # The hooks fire on every input-batch add/remove (a request is
                # evicted + re-added across steps under b1-prefill / partial
                # scheduling), so a req_id recurs. Keep the FIRST admit (true
                # arrival) and the LAST completion (final out_len).
                if rid not in admits:
                    admits[rid] = e
            elif kind == "request_completed":
                completes[rid] = e
            elif kind == "step_snapshot":
                snapshots.append(e)
    snapshots.sort(key=lambda s: s.get("ts", 0))
    return admits, completes, snapshots


def _classify(slot):
    state = slot.get("state")
    if state == "PREFILL":
        return PREFILL
    if state == "DECODE":
        return STALLED if slot.get("scheduled") == 0 else DECODE
    return (state or "").lower()


def build_run(admits, completes, snapshots):
    """Assemble a time-series run model from the raw events."""
    all_ts = [e["ts"] for e in admits.values()] + [e["ts"] for e in completes.values()]
    all_ts += [s["ts"] for s in snapshots]
    if not all_ts:
        return None
    t0, t_end = min(all_ts), max(all_ts)

    # num_slots: prefer the engine-reported value (snapshots). Without snapshots
    # it can only be inferred as a LOWER BOUND from slot indices seen across all
    # events -- the true batch size needs TT_INSTRUMENT_SNAPSHOTS_JSONL=1.
    num_slots = 0
    num_slots_exact = False
    for s in snapshots:
        if s.get("num_slots"):
            num_slots = max(num_slots, s["num_slots"])
            num_slots_exact = True
    if not num_slots:
        idxs = [
            e.get("slot_idx")
            for e in list(admits.values()) + list(completes.values())
            if e.get("slot_idx") is not None
        ]
        num_slots = (max(idxs) + 1) if idxs else 1

    # Per-request records (joined admit + completion).
    requests = {}
    for rid, a in admits.items():
        requests[rid] = {
            "req_id": rid,
            "slot_idx": a.get("slot_idx"),
            "isl": a.get("isl"),
            "arrival": a["ts"],
            "completion": None,
            "out_len": None,
            "ttft": None,
            "mean_rate": None,
            "finish_reason": None,
        }
    for rid, c in completes.items():
        r = requests.setdefault(
            rid,
            {
                "req_id": rid,
                "slot_idx": c.get("slot_idx"),
                "isl": c.get("isl"),
                "arrival": None,
            },
        )
        r["completion"] = c["ts"]
        r["out_len"] = c.get("out_len")
        r["ttft"] = c.get("ttft")
        r["mean_rate"] = c.get("mean_rate")
        r["finish_reason"] = c.get("finish_reason")

    # Per-slot state segments + step-kind bands from the snapshot series.
    segments, steps = [], []
    open_seg = {}  # slot_idx -> dict being extended

    def close(slot, t1):
        seg = open_seg.pop(slot, None)
        if seg is not None:
            seg["t1"] = t1
            segments.append(seg)

    for i, snap in enumerate(snapshots):
        ts = snap["ts"]
        nxt = snapshots[i + 1]["ts"] if i + 1 < len(snapshots) else t_end
        steps.append({"t0": ts - t0, "t1": nxt - t0, "kind": snap.get("step_kind")})
        seen = set()
        for slot in snap.get("slots", []):
            si = slot.get("slot_idx")
            if si is None:
                continue
            seen.add(si)
            cls = _classify(slot)
            rid = str(slot.get("req_id"))
            cur = open_seg.get(si)
            if cur and (cur["cls"] != cls or cur["req"] != rid):
                close(si, ts - t0)
                cur = None
            if cur is None:
                open_seg[si] = {
                    "slot": si,
                    "t0": ts - t0,
                    "t1": None,
                    "cls": cls,
                    "req": rid,
                }
        # Slots absent this step are free -> close their open segment.
        for si in list(open_seg):
            if si not in seen:
                close(si, ts - t0)
    for si in list(open_seg):
        close(si, t_end - t0)

    return {
        "t0": t0,
        "t_end": t_end,
        "duration": t_end - t0,
        "num_slots": num_slots,
        "num_slots_exact": num_slots_exact,
        "requests": requests,
        "segments": segments,
        "steps": steps,
        "snapshots": snapshots,
        "has_snapshots": bool(snapshots),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _pct(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "p50": _pct(vals, 50),
        "p90": _pct(vals, 90),
        "max": max(vals),
    }


def compute_metrics(run):
    reqs = list(run["requests"].values())
    dur = run["duration"] or 1e-9
    completed = [r for r in reqs if r["completion"] is not None]
    total_out = sum(r["out_len"] or 0 for r in completed)

    m = {
        "duration_s": run["duration"],
        "num_slots": run["num_slots"],
        "num_slots_exact": run["num_slots_exact"],
        "requests_admitted": sum(1 for r in reqs if r["arrival"] is not None),
        "requests_completed": len(completed),
        "total_output_tokens": total_out,
        "throughput_tok_s": total_out / dur,
        "ttft_s": _stat([r["ttft"] for r in reqs]),
        "isl": _stat([r["isl"] for r in reqs]),
        "osl": _stat([r["out_len"] for r in completed]),
        "decode_rate_tok_s": _stat([r["mean_rate"] for r in reqs]),
        "e2e_latency_s": _stat(
            [
                r["completion"] - r["arrival"]
                for r in completed
                if r["arrival"] is not None
            ]
        ),
    }

    # Concurrency / overlap from request intervals (works without snapshots).
    pts = []
    for r in reqs:
        a = r["arrival"]
        c = r["completion"] if r["completion"] is not None else run["t_end"]
        if a is not None:
            pts.append((a, 1))
            pts.append((c, -1))
    pts.sort()
    cur = mx = 0
    integral = multi = prev = 0.0
    for t, delta in pts:
        if prev:
            integral += cur * (t - prev)
            if cur >= 2:
                multi += t - prev
        cur += delta
        mx = max(mx, cur)
        prev = t
    m["concurrency"] = {
        "mean": integral / dur,
        "max": mx,
        "pct_time_multi": 100.0 * multi / dur,
    }

    # Snapshot-only: where slot-time went, stall cost, step-kind wall time.
    if run["has_snapshots"]:
        cls_time = {PREFILL: 0.0, DECODE: 0.0, STALLED: 0.0}
        for seg in run["segments"]:
            if seg["cls"] in cls_time and seg["t1"] is not None:
                cls_time[seg["cls"]] += seg["t1"] - seg["t0"]
        m["slot_seconds"] = cls_time  # summed across slots
        kind_time = {}
        for s in run["steps"]:
            kind_time[s["kind"]] = kind_time.get(s["kind"], 0.0) + (s["t1"] - s["t0"])
        m["step_kind_seconds"] = kind_time
        occ = [s.get("num_running", 0) for s in run["snapshots"]]
        m["occupancy"] = {
            "mean": (sum(occ) / len(occ)) if occ else 0,
            "max": max(occ) if occ else 0,
            "of_slots": run["num_slots"],
        }
    return m


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #
def _row(label, st, unit=""):
    if st["n"] == 0:
        return f"| {label} | - | - | - | - |"
    f = lambda v: f"{v:.2f}{unit}" if v is not None else "-"
    return f"| {label} | {f(st['mean'])} | {f(st['p50'])} | {f(st['p90'])} | {f(st['max'])} |"


def render_markdown(run, m):
    o = []
    o.append("# TT vLLM run analysis\n")
    slots_str = (
        f"{m['num_slots']}" if m["num_slots_exact"] else f"≥{m['num_slots']} (inferred)"
    )
    o.append(f"- duration: **{m['duration_s']:.1f}s**, slots: **{slots_str}**")
    o.append(
        f"- requests: {m['requests_admitted']} admitted, "
        f"{m['requests_completed']} completed"
    )
    o.append(
        f"- output: {m['total_output_tokens']} tokens, "
        f"**{m['throughput_tok_s']:.1f} tok/s** aggregate"
    )
    c = m["concurrency"]
    o.append(
        f"- concurrency: mean {c['mean']:.2f}, peak {c['max']}, "
        f"{c['pct_time_multi']:.0f}% of time with ≥2 active"
    )
    if not run["has_snapshots"]:
        o.append(
            "\n> events-only log (no step_snapshots): per-slot state, stall "
            "cost, and step breakdown unavailable. Re-run the server with "
            "`TT_INSTRUMENT_SNAPSHOTS_JSONL=1` for the full timeline."
        )
    o.append("\n## Per-request (mean / p50 / p90 / max)\n")
    o.append("| metric | mean | p50 | p90 | max |")
    o.append("|---|---|---|---|---|")
    o.append(_row("TTFT", m["ttft_s"], "s"))
    o.append(_row("decode rate (tok/s)", m["decode_rate_tok_s"]))
    o.append(_row("ISL (tok)", m["isl"]))
    o.append(_row("OSL (tok)", m["osl"]))
    o.append(_row("end-to-end latency", m["e2e_latency_s"], "s"))

    if run["has_snapshots"]:
        o.append("\n## Where slot-time went (summed across slots)\n")
        ss = m["slot_seconds"]
        tot = sum(ss.values()) or 1e-9
        o.append("| state | slot-seconds | share |")
        o.append("|---|---|---|")
        for k in (PREFILL, DECODE, STALLED):
            o.append(f"| {k} | {ss[k]:.1f} | {100*ss[k]/tot:.0f}% |")
        o.append(
            f"\n- **stall cost**: {ss[STALLED]:.1f} slot-seconds of decode "
            "lost to interference (a peer prefilling)"
        )
        occ = m["occupancy"]
        o.append(
            f"- occupancy: mean {occ['mean']:.1f} / peak {occ['max']} "
            f"of {occ['of_slots']} slots"
        )
        o.append("\n## Step kind (wall time)\n")
        kt = m["step_kind_seconds"]
        tot = sum(kt.values()) or 1e-9
        o.append("| step kind | seconds | share |")
        o.append("|---|---|---|")
        for k, v in sorted(kt.items(), key=lambda x: -x[1]):
            o.append(f"| {k} | {v:.1f} | {100*v/tot:.0f}% |")
    return "\n".join(o) + "\n"


# --------------------------------------------------------------------------- #
# HTML timeline (self-contained: inline SVG + JS wheel-zoom / drag-pan)
# --------------------------------------------------------------------------- #
_HTML_JS = r"""
const PPS0 = Math.max(2, Math.min(40, 1100 / (META.duration || 1)));
let pps = PPS0;
const ROW = 16, COLORS = META.colors;
const svg = document.getElementById('tl');
const tip = document.getElementById('tip');
function render() {
  const W = Math.max(200, META.duration * pps), H = META.num_slots * ROW + 24;
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  let s = '';
  // step-kind band
  for (const k of STEPS) {
    const c = {prefill:COLORS.prefill, decode:COLORS.decode, mixed:'#8e44ad'}[k.kind] || '#444';
    s += `<rect x="${k.t0*pps}" y="0" width="${Math.max(0.5,(k.t1-k.t0)*pps)}" height="6" fill="${c}" opacity="0.5"/>`;
  }
  // slot row separators
  for (let i=0;i<META.num_slots;i++){ s += `<line x1="0" y1="${10+i*ROW}" x2="${W}" y2="${10+i*ROW}" stroke="#eee"/>`; }
  // segments (or request bars when events-only)
  const data = SEGMENTS.length ? SEGMENTS : BARS;
  for (const seg of data) {
    const y = 10 + seg.slot*ROW + 1, x = seg.t0*pps, w = Math.max(1,(seg.t1-seg.t0)*pps);
    const color = COLORS[seg.cls] || COLORS.request;
    const meta = `${seg.req||''} ${seg.cls||'req'} ${(seg.t0).toFixed(2)}-${(seg.t1).toFixed(2)}s`;
    s += `<rect x="${x}" y="${y}" width="${w}" height="${ROW-2}" fill="${color}" data-m="${meta}"/>`;
  }
  svg.innerHTML = s;
}
svg.addEventListener('mousemove', e => {
  const m = e.target && e.target.getAttribute && e.target.getAttribute('data-m');
  if (m){ tip.style.display='block'; tip.style.left=(e.pageX+12)+'px'; tip.style.top=(e.pageY+12)+'px'; tip.textContent=m; }
  else tip.style.display='none';
});
const wrap = document.getElementById('wrap');
wrap.addEventListener('wheel', e => {
  if (!e.ctrlKey && !e.shiftKey && Math.abs(e.deltaY) < Math.abs(e.deltaX)) return;
  e.preventDefault();
  const tAt = (wrap.scrollLeft + e.clientX - wrap.getBoundingClientRect().left) / pps;
  pps *= (e.deltaY < 0 ? 1.2 : 1/1.2); pps = Math.max(0.5, Math.min(2000, pps));
  render(); wrap.scrollLeft = tAt * pps - (e.clientX - wrap.getBoundingClientRect().left);
}, {passive:false});
let drag=null;
wrap.addEventListener('mousedown', e => { drag={x:e.clientX, sl:wrap.scrollLeft}; });
window.addEventListener('mouseup', () => drag=null);
window.addEventListener('mousemove', e => { if(drag) wrap.scrollLeft = drag.sl-(e.clientX-drag.x); });
document.getElementById('zin').onclick=()=>{pps*=1.5;render();};
document.getElementById('zout').onclick=()=>{pps/=1.5;render();};
document.getElementById('zrst').onclick=()=>{pps=PPS0;render();wrap.scrollLeft=0;};
render();
"""


def _md_table_to_html(md):
    # Minimal markdown -> HTML for the summary block (headers, lists, tables).
    out, in_tbl = [], False
    for ln in md.splitlines():
        if ln.startswith("| "):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue  # separator row
            tag = "th" if not in_tbl else "td"
            if not in_tbl:
                out.append("<table>")
                in_tbl = True
            out.append(
                "<tr>"
                + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells)
                + "</tr>"
            )
            continue
        if in_tbl:
            out.append("</table>")
            in_tbl = False
        if ln.startswith("# "):
            out.append(f"<h1>{html.escape(ln[2:])}</h1>")
        elif ln.startswith("## "):
            out.append(f"<h2>{html.escape(ln[3:])}</h2>")
        elif ln.startswith(("- ", "> ")):
            out.append(f"<p>{html.escape(ln[2:])}</p>")
        elif ln.strip():
            out.append(f"<p>{html.escape(ln)}</p>")
    if in_tbl:
        out.append("</table>")
    return "\n".join(out)


def render_html(run, m, md):
    bars = [
        {
            "slot": r["slot_idx"] or 0,
            "t0": (r["arrival"] - run["t0"]) if r["arrival"] else 0,
            "t1": ((r["completion"] or run["t_end"]) - run["t0"]),
            "req": r["req_id"],
            "cls": "request",
        }
        for r in run["requests"].values()
        if r["slot_idx"] is not None and r["arrival"] is not None
    ]
    payload = {
        "META": {
            "duration": run["duration"],
            "num_slots": run["num_slots"],
            "colors": _COLORS,
        },
        "SEGMENTS": [
            {
                "slot": s["slot"],
                "t0": s["t0"],
                "t1": s["t1"] or s["t0"],
                "cls": s["cls"],
                "req": s["req"],
            }
            for s in run["segments"]
        ],
        "BARS": bars,
        "STEPS": [s for s in run["steps"] if s["kind"]],
    }
    data_js = "".join(f"const {k}={json.dumps(v)};\n" for k, v in payload.items())
    summary_html = _md_table_to_html(md)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>TT vLLM run</title><style>"
        "body{font-family:system-ui,sans-serif;margin:16px;color:#222}"
        "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:2px 8px;font-size:13px}"
        "th{background:#f4f4f4}#wrap{overflow-x:auto;border:1px solid #ccc;cursor:grab}"
        "#tip{position:absolute;display:none;background:#000;color:#fff;padding:2px 6px;"
        "border-radius:3px;font-size:12px;pointer-events:none;z-index:9}"
        "button{margin-right:6px}.legend span{margin-right:12px;font-size:13px}"
        "</style></head><body>" + summary_html + "<h2>Slot timeline</h2>"
        "<div class='legend'>"
        "<span style='color:#d4a017'>■ prefill</span>"
        "<span style='color:#2e8b57'>■ decode</span>"
        "<span style='color:#c0392b'>■ stalled</span>"
        "<span style='color:#4a90d2'>■ request (events-only)</span>"
        "&nbsp; <button id='zin'>zoom +</button><button id='zout'>zoom −</button>"
        "<button id='zrst'>reset</button> "
        "<small>ctrl/shift+wheel to zoom, drag to pan</small></div>"
        "<div id='wrap'><svg id='tl' xmlns='http://www.w3.org/2000/svg'></svg></div>"
        "<div id='tip'></div>"
        "<script>" + data_js + _HTML_JS + "</script></body></html>"
    )


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Offline analysis of a TT vLLM run")
    p.add_argument("--dir", help="telemetry dir (TT_INSTRUMENT_DIR)")
    p.add_argument("--events", help="path to events.jsonl (overrides --dir)")
    p.add_argument("--html", help="write a self-contained timeline HTML here")
    p.add_argument("--json", help="write parsed metrics JSON here")
    p.add_argument("--md", help="write the markdown summary here (default: stdout)")
    args = p.parse_args()

    events_path = args.events or (
        os.path.join(args.dir, "events.jsonl") if args.dir else None
    )
    if not events_path or not os.path.exists(events_path):
        p.error("need --events PATH or --dir DIR containing events.jsonl")

    run = build_run(*parse_events(events_path))
    if run is None:
        print("No events found in the log.", file=sys.stderr)
        sys.exit(1)
    metrics = compute_metrics(run)
    md = render_markdown(run, metrics)

    if args.md:
        open(args.md, "w").write(md)
    else:
        sys.stdout.write(md)
    if args.json:
        open(args.json, "w").write(json.dumps(metrics, indent=2))
    if args.html:
        open(args.html, "w").write(render_html(run, metrics, md))
        print(f"\nWrote timeline -> {args.html}", file=sys.stderr)


if __name__ == "__main__":
    main()
