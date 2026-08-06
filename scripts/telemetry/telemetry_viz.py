# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Visualize vLLM serving telemetry.

Consumes the JSON-lines sinks written by vllm_tt/telemetry.py
(scheduler.jsonl, runner.jsonl, runner_snapshot.json) and renders a single
self-contained interactive HTML dashboard: slot-occupancy timeline,
decode-rate / KV / batch-utilization overlays, a per-request Gantt, and an
event strip (prefill-stalled-decode, preemption, watermark rejects, b1 caps).

Two modes:

  report  Parse a finished run's sinks and write a standalone HTML file
          (data embedded; no server needed to view).

            python scripts/telemetry/telemetry_viz.py report [--dir tt_telemetry] \
                [--out tt_telemetry/report.html]

  live    Serve the same dashboard on localhost and poll the sinks while a
          model is serving, so slot state / decode rate update in place.

            python scripts/telemetry/telemetry_viz.py live [--dir tt_telemetry] \
                [--port 8009] [--interval-ms 500]

The dashboard is dependency-free (stdlib only) and theme-aware (light/dark).
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCHEDULER_FILE = "scheduler.jsonl"
RUNNER_FILE = "runner.jsonl"
SNAPSHOT_FILE = "runner_snapshot.json"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A live tail may catch a half-written final line; skip it.
                continue
    return out


def load(directory):
    """Load and split the telemetry sinks into the shape the page renders from."""
    d = Path(directory)
    scheduler = _read_jsonl(d / SCHEDULER_FILE)
    runner = _read_jsonl(d / RUNNER_FILE)
    snapshot = None
    snap_path = d / SNAPSHOT_FILE
    if snap_path.exists():
        try:
            snapshot = json.loads(snap_path.read_text())
        except (json.JSONDecodeError, OSError):
            snapshot = None

    runner_steps = [r for r in runner if r.get("event") == "step"]
    admits = [r for r in runner if r.get("event") == "request_admitted"]
    completes = [r for r in runner if r.get("event") == "request_completed"]
    return {
        "scheduler": scheduler,
        "runner_steps": runner_steps,
        "admits": admits,
        "completes": completes,
        "snapshot": snapshot,
    }


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
def render_html(data, live=False, interval_ms=500, body_only=False):
    payload = json.dumps(data, separators=(",", ":"))
    page = (
        _template()
        .replace("/*__LIVE__*/", "true" if live else "false")
        .replace("/*__INTERVAL__*/", str(int(interval_ms)))
        .replace("/*__DATA__*/", payload if not live else "null")
    )
    if not body_only:
        return page
    # Emit page content only (style + body markup + script), no document
    # scaffolding: suitable for a host that wraps it in its own <head>/<body>
    # skeleton (e.g. a claude.ai Artifact). The <title> is dropped (the host
    # supplies it); the <style> block is kept (inline CSS is still required).
    start = page.index("<style>")
    end = page.index("</script>") + len("</script>")
    chunk = page[start:end]
    return chunk.replace("</head>", "").replace("<body>", "").strip()


_TEMPLATE_PATH = Path(__file__).resolve().parent / "telemetry_viz_template.html"
_TEMPLATE = None


def _template():
    """Read (and cache) the page template shipped beside this script."""
    global _TEMPLATE
    if _TEMPLATE is None:
        try:
            _TEMPLATE = _TEMPLATE_PATH.read_text()
        except OSError as e:
            raise SystemExit(
                f"telemetry_viz: cannot read page template " f"{_TEMPLATE_PATH}: {e}"
            )
    return _TEMPLATE


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def cmd_report(args):
    data = load(args.dir)
    n_sched = len(data["scheduler"])
    n_run = len(data["runner_steps"])
    if n_sched == 0 and n_run == 0:
        print(
            f"No telemetry found in {args.dir!r} "
            f"(looked for {SCHEDULER_FILE} / {RUNNER_FILE}). "
            f"Run a serving job with TTXLA_TELEMETRY=1 first.",
            file=sys.stderr,
        )
        return 1
    out = args.out or os.path.join(args.dir, "report.html")
    Path(out).write_text(render_html(data, live=False, body_only=args.body_only))
    print(f"Wrote {out}")
    print(f"  scheduler steps: {n_sched}")
    print(f"  runner steps:    {n_run}")
    print(
        f"  requests:        {len(data['admits'])} admitted, "
        f"{len(data['completes'])} completed"
    )
    print(f"Open it: file://{os.path.abspath(out)}")
    return 0


def cmd_live(args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    directory = args.dir

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                html = render_html(None, live=True, interval_ms=args.interval_ms)
                self._send(html.encode(), "text/html; charset=utf-8")
            elif path == "/data.json":
                self._send(json.dumps(load(directory)).encode(), "application/json")
            else:
                self.send_error(404)

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        print(
            f"error: cannot bind to port {args.port}: {e}\n"
            f"Is another telemetry_viz process already running?  "
            f"Try: lsof -ti :{args.port} | xargs kill",
            file=sys.stderr,
        )
        return 1
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving telemetry dashboard for {directory!r} at {url}")
    print(f"Polling every {args.interval_ms} ms. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        srv.server_close()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser(
        "report", help="write a standalone HTML report from a finished run"
    )
    pr.add_argument(
        "--dir",
        default="tt_telemetry",
        help="telemetry directory (default: tt_telemetry)",
    )
    pr.add_argument(
        "--out", default=None, help="output HTML path (default: <dir>/report.html)"
    )
    pr.add_argument(
        "--body-only",
        action="store_true",
        help="emit page content only (no <html>/<head>/<body>), for embedding in "
        "a host that supplies the document skeleton",
    )
    pr.set_defaults(func=cmd_report)

    pl = sub.add_parser("live", help="serve a live-updating dashboard on localhost")
    pl.add_argument(
        "--dir",
        default="tt_telemetry",
        help="telemetry directory (default: tt_telemetry)",
    )
    pl.add_argument("--port", type=int, default=8009, help="port (default: 8009)")
    pl.add_argument(
        "--interval-ms",
        type=int,
        default=500,
        help="browser poll interval (default: 500)",
    )
    pl.set_defaults(func=cmd_live)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
