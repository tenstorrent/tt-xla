#!/usr/bin/env python3
"""Enforce the bisection log-cache retention rule.

Raw CI job logs under bisection/logs/run_<id>/ are a CACHE: they are only needed
while a run is being analysed, and they dominate the directory size (~5 MB per
nightly). The durable artifacts -- bisected.json (the ledger),
run_<id>_failures{,_filtered}.json and nightly_<id>_failures_table.md -- are
small and are NEVER touched by this script.

Retention: keep raw logs for nightlies from the last RETENTION_DAYS days
(default 4). Everything older is pruned, as is any log directory with no
resolvable run date (those are boundary-search leftovers with no analysis JSON).

Caveat worth knowing: GitHub expires job logs, so pruning is a one-way door --
an expired log cannot be re-downloaded (the API returns a BlobNotFound XML
payload, which earlier versions of the downloader happily saved *as* the log).
Only the failures JSONs let you trace signatures across older nightlies, which
is why they are kept indefinitely.

Usage:
    python3 .claude/scripts/prune_logs.py              # dry run, shows what would go
    python3 .claude/scripts/prune_logs.py --apply      # actually delete
    python3 .claude/scripts/prune_logs.py --apply --days 7
    python3 .claude/scripts/prune_logs.py --apply --keep-min 2
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone

RETENTION_DAYS = 4
KEEP_MIN = 1  # never leave the cache completely empty


def repo_root() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        print("ERROR: not inside a git repository", file=sys.stderr)
        raise SystemExit(1)


def run_date_for(bisection: str, run_id: str, logs_dir: str):
    """Resolve a run's date from the analysis JSON, then the cache index."""
    fpath = os.path.join(bisection, f"run_{run_id}_failures.json")
    if os.path.exists(fpath):
        try:
            with open(fpath) as fh:
                raw = json.load(fh).get("run_date", "")
            if raw:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    ipath = os.path.join(logs_dir, "index.json")
    if os.path.exists(ipath):
        try:
            with open(ipath) as fh:
                raw = json.load(fh).get(f"run_{run_id}", {}).get("date", "")
            if raw:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    return None


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def find_error_payloads(logs_dir: str):
    """Job logs that are actually API error payloads, not logs. Always junk."""
    bad = []
    for root, _, files in os.walk(logs_dir):
        for f in files:
            if not (f.startswith("job_") and f.endswith(".txt")):
                continue
            p = os.path.join(root, f)
            try:
                if os.path.getsize(p) > 2048:
                    continue
                with open(p, "rb") as fh:
                    head = fh.read(80).decode("utf-8", "replace")
                if "<Error>" in head or "BlobNotFound" in head:
                    bad.append(p)
            except OSError:
                pass
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument("--days", type=int, default=RETENTION_DAYS,
                    help=f"retention window in days (default {RETENTION_DAYS})")
    ap.add_argument("--keep-min", type=int, default=KEEP_MIN,
                    help=f"always keep at least this many newest runs (default {KEEP_MIN})")
    args = ap.parse_args()

    root = repo_root()
    bisection = os.path.join(root, "bisection")
    logs_dir = os.path.join(bisection, "logs")
    if not os.path.isdir(logs_dir):
        print(f"nothing to do: {logs_dir} does not exist")
        return 0

    today = date.today()
    entries = []
    for name in sorted(os.listdir(logs_dir)):
        path = os.path.join(logs_dir, name)
        if not os.path.isdir(path):
            continue
        m = re.fullmatch(r"run_(\d+)", name)
        if not m:
            continue
        d = run_date_for(bisection, m.group(1), logs_dir)
        age = (today - d).days if d else None
        entries.append({"name": name, "path": path, "date": d, "age": age,
                        "mb": dir_size_mb(path)})

    # Newest first; undated (boundary-search leftovers) sort last.
    entries.sort(key=lambda e: e["date"] or date.min, reverse=True)
    for i, e in enumerate(entries):
        stale = e["age"] is None or e["age"] >= args.days
        e["keep"] = (i < args.keep_min) or not stale
        e["why"] = ("newest, below --keep-min" if i < args.keep_min and stale
                    else "within window" if not stale
                    else "no resolvable run date" if e["age"] is None
                    else f"{e['age']}d old")

    payloads = find_error_payloads(logs_dir)

    print(f"retention: {args.days} days (keep-min {args.keep_min}) - today {today}\n")
    for e in entries:
        print(f"  {'KEEP ' if e['keep'] else 'PRUNE'}  {e['name']:<22} "
              f"{str(e['date'] or '?'):<12} {e['mb']:7.1f} MB   {e['why']}")
    if payloads:
        print(f"\n  {len(payloads)} API error payload(s) saved as logs (always pruned):")
        for p in payloads:
            print(f"    {os.path.relpath(p, root)}")

    doomed = [e for e in entries if not e["keep"]]
    freed = sum(e["mb"] for e in doomed)
    print(f"\n{'would free' if not args.apply else 'freeing'}: {freed:.1f} MB "
          f"from {len(doomed)} run(s)")

    if not args.apply:
        print("\ndry run - pass --apply to delete")
        return 0

    for p in payloads:
        os.remove(p)
    for e in doomed:
        shutil.rmtree(e["path"])
        print(f"  removed {e['name']}")

    # Keep the cache index consistent with what actually survives on disk.
    ipath = os.path.join(logs_dir, "index.json")
    if os.path.exists(ipath):
        try:
            with open(ipath) as fh:
                idx = json.load(fh)
            kept = {k: v for k, v in idx.items()
                    if os.path.isdir(os.path.join(logs_dir, k))}
            if len(kept) != len(idx):
                with open(ipath, "w") as fh:
                    json.dump(kept, fh, indent=1)
                print(f"  index.json: dropped {len(idx) - len(kept)} stale entry(ies)")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: could not tidy index.json: {exc}", file=sys.stderr)

    print(f"\ndone - freed {freed:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
