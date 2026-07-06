#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Persistent bisection ledger for the nightly auto-bisect skill.

Tracks, per (test_id, machine_type), what has already happened so re-running the
skill on later nightlies does NOT re-bisect regressions we've already resolved,
and so "bisection pending" survives across runs.

Ledger file (default `bisection/bisected.json`):
{
  "updated_at": "<UTC ISO>",
  "entries": {
    "<test_id>||<machine_type>": {
      "test_id", "machine_type",
      "status": "bisected|pending|group_only|timeout|checkpoint_drift",
      "blame_sha", "blame_pr",              # when status == bisected
      "first_bad_run_id", "first_bad_sha", "first_bad_date",
      "last_good_sha", "known_error", "notes",
      "last_updated": "<UTC ISO>"
    }, ...
  }
}

STATUS meanings
  bisected        - culprit commit found (blame_sha/blame_pr recorded); skip on re-run
  pending         - a real regression whose bisection has NOT been done yet
  group_only      - passes standalone, fails only in the nightly group (pollution/ordering); not code-bisectable in isolation
  timeout         - job timed out / hung; no reliable good/bad signal
  checkpoint_drift- PCC-style change likely from an HF checkpoint update, not a commit

Usage:
  python3 .claude/scripts/ledger.py get  --test <id> --machine <m> [--ledger PATH]
  python3 .claude/scripts/ledger.py set  --test <id> --machine <m> --status <s> \
          [--blame-sha S] [--blame-pr URL] [--first-bad-run-id N] [--first-bad-sha S] \
          [--first-bad-date D] [--last-good-sha S] [--known-error E] [--notes N] \
          [--field KEY=VALUE ...] [--ledger PATH]
  python3 .claude/scripts/ledger.py list [--status pending] [--ledger PATH]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("bisected", "pending", "group_only", "timeout", "checkpoint_drift")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except Exception:
        return Path.cwd()


def default_ledger_path():
    return _repo_root() / "bisection" / "bisected.json"


def key(test_id: str, machine_type: str) -> str:
    return f"{test_id}||{machine_type}"


def load(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"updated_at": None, "entries": {}}


def save(path: Path, ledger: dict):
    ledger["updated_at"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True))


def lookup(ledger: dict, test_id: str, machine_type: str):
    return ledger.get("entries", {}).get(key(test_id, machine_type))


def upsert(ledger: dict, test_id: str, machine_type: str, **fields):
    entries = ledger.setdefault("entries", {})
    k = key(test_id, machine_type)
    entry = entries.get(k, {})
    entry.update({"test_id": test_id, "machine_type": machine_type})
    for name, val in fields.items():
        if val is not None:
            entry[name] = val
    entry["last_updated"] = _now()
    entries[k] = entry
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_get(args, path):
    entry = lookup(load(path), args.test, args.machine)
    print(json.dumps(entry, indent=2) if entry else "{}")
    return 0


def _cmd_set(args, path):
    if args.status not in STATUSES:
        print(f"ERROR: --status must be one of {STATUSES}", file=sys.stderr)
        return 2
    ledger = load(path)
    fields = {
        "status": args.status,
        "blame_sha": args.blame_sha,
        "blame_pr": args.blame_pr,
        "first_bad_run_id": args.first_bad_run_id,
        "first_bad_sha": args.first_bad_sha,
        "first_bad_date": args.first_bad_date,
        "last_good_sha": args.last_good_sha,
        "known_error": args.known_error,
        "notes": args.notes,
    }
    for kv in args.field or []:
        if "=" not in kv:
            print(f"ERROR: --field expects KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 2
        fk, fv = kv.split("=", 1)
        fields[fk] = fv
    entry = upsert(ledger, args.test, args.machine, **fields)
    save(path, ledger)
    print(json.dumps(entry, indent=2))
    return 0


def _cmd_list(args, path):
    ledger = load(path)
    rows = list(ledger.get("entries", {}).values())
    if args.status:
        rows = [r for r in rows if r.get("status") == args.status]
    print(json.dumps(rows, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ledger", help="Ledger path (default: <repo>/bisection/bisected.json)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser(
        "get", help="Print the entry for (test, machine), or {} if absent"
    )
    g.add_argument("--test", required=True)
    g.add_argument("--machine", required=True)

    s = sub.add_parser("set", help="Create/update an entry")
    s.add_argument("--test", required=True)
    s.add_argument("--machine", required=True)
    s.add_argument("--status", required=True)
    s.add_argument("--blame-sha")
    s.add_argument("--blame-pr")
    s.add_argument("--first-bad-run-id")
    s.add_argument("--first-bad-sha")
    s.add_argument("--first-bad-date")
    s.add_argument("--last-good-sha")
    s.add_argument("--known-error")
    s.add_argument("--notes")
    s.add_argument("--field", action="append", help="Extra KEY=VALUE (repeatable)")

    l = sub.add_parser("list", help="List entries, optionally filtered by --status")
    l.add_argument("--status", choices=STATUSES)

    args = ap.parse_args(argv)
    path = Path(args.ledger) if args.ledger else default_ledger_path()

    return {"get": _cmd_get, "set": _cmd_set, "list": _cmd_list}[args.cmd](args, path)


if __name__ == "__main__":
    raise SystemExit(main())
