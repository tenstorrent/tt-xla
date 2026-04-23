#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
compare_perf.py - Compare model SPS from a GitHub Actions logs archive
against latest_perf.csv baseline.

Usage (from project root):
    # Download logs from a run (needs GH_TOKEN / GITHUB_TOKEN / ~/.ghtoken)
    python3 .claude/skills/perf-compare/scripts/compare_perf.py --run-id 24733959318

    # Use a pre-downloaded logs zip
    python3 .claude/skills/perf-compare/scripts/compare_perf.py --logs-zip /path/to/logs.zip

    # Use a pre-extracted logs directory
    python3 .claude/skills/perf-compare/scripts/compare_perf.py --logs-dir /path/to/logs/
"""

import argparse
import csv
import os
import re
import subprocess
import zipfile
from pathlib import Path


# ── Log timestamp prefix ───────────────────────────────────────────────────────
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ")

def strip_ts(line: str) -> str:
    """Remove GitHub Actions timestamp prefix from a log line."""
    return _TS_RE.sub("", line)


# ── Log file filtering ─────────────────────────────────────────────────────────
def is_perf_log(filename: str) -> bool:
    """True for files like '8_run-perf-benchmarks _ ... _ perf bert (n150-perf).txt'"""
    return "run-perf-benchmarks" in filename and filename.endswith(".txt")


# ── Per-log parsing ────────────────────────────────────────────────────────────
_PYTEST_CMD_RE = re.compile(r"pytest\s.*?::(\w+)\s")
_SPS_RE        = re.compile(r"\|\s*Sample per second:\s*([\d.]+)")


def parse_log(path: Path) -> dict | None:
    """Parse one perf log file and return a result dict, or None if model name not found."""
    model_name = None
    sps = None

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  WARNING: Cannot read {path.name}: {e}")
        return None

    for raw_line in text.splitlines():
        line = strip_ts(raw_line).rstrip()

        # Extract model name from the pytest command line (e.g. ::test_falcon3_1b)
        if model_name is None and "pytest" in line and "::" in line:
            m = _PYTEST_CMD_RE.search(line)
            if m:
                test_fn = m.group(1)  # e.g. "test_falcon3_1b"
                if test_fn.startswith("test_"):
                    model_name = test_fn[5:]  # strip "test_" → "falcon3_1b"

        # SPS
        m = _SPS_RE.search(line)
        if m:
            sps = float(m.group(1))

    if model_name is None:
        return None  # not a perf benchmark log we can use

    return {
        "model":   model_name,
        "sps_new": sps,
    }


# ── Baseline CSV ───────────────────────────────────────────────────────────────
_HTML_NUM_RE = re.compile(r">([\d.]+)<")

def _extract_numeric(value: str) -> float | None:
    if not value or not value.strip():
        return None
    m = _HTML_NUM_RE.search(value)
    if m:
        try: return float(m.group(1))
        except ValueError: pass
    try: return float(value.strip())
    except ValueError: return None


def load_baseline(csv_path: str) -> dict[str, float]:
    """model short-name → current SPS from latest_perf.csv."""
    baseline: dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row.get("model", "").strip()
            sps   = _extract_numeric(row.get("samples per second", ""))
            if model and sps is not None:
                baseline[model] = sps
    return baseline


# ── Logs download ──────────────────────────────────────────────────────────────
def _get_token() -> str | None:
    """Try the same token sources as download_artifacts.py."""
    tok_file = Path.home() / ".ghtoken"
    if tok_file.exists():
        tok = tok_file.read_text().strip()
        if tok:
            return tok
    for env in ("GITHUB_TOKEN", "GH_TOKEN"):
        tok = os.environ.get(env, "")
        if tok:
            return tok
    try:
        tok = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
        if tok:
            return tok
    except Exception:
        pass
    return None


def download_logs_zip(run_id: str, repo: str, dest_zip: str) -> None:
    """Download the run logs archive via GitHub REST API."""
    import urllib.request
    import urllib.error

    token = _get_token()
    if not token:
        raise RuntimeError(
            "No GitHub token found. Set GH_TOKEN / GITHUB_TOKEN, put a token in "
            "~/.ghtoken, or run: gh auth login (choose HTTPS)"
        )

    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs"
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    })
    print(f"Downloading logs archive for run {run_id} ...")
    try:
        with urllib.request.urlopen(req) as resp:
            Path(dest_zip).write_bytes(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GitHub API error {e.code}: {e.reason}") from e
    print(f"Saved to {dest_zip}")


# ── Main ───────────────────────────────────────────────────────────────────────
def load_log_results(logs_dir: str) -> dict[str, dict]:
    """Walk logs_dir for perf log files and return model → result dict."""
    results: dict[str, dict] = {}
    log_files = [
        p for p in Path(logs_dir).rglob("*.txt")
        if is_perf_log(p.name)
    ]
    print(f"  Found {len(log_files)} perf log file(s).")
    for path in log_files:
        r = parse_log(path)
        if r is None:
            continue
        model = r["model"]
        if model in results:
            print(f"  WARNING: duplicate model '{model}', keeping first occurrence")
            continue
        results[model] = r
    return results


def build_rows(
    baseline: dict[str, float],
    log_results: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    tp_rows: list[dict] = []
    non_tp_rows: list[dict] = []
    matched = 0

    for model, current_sps in baseline.items():
        if model not in log_results:
            continue
        matched += 1
        r       = log_results[model]
        new_sps = r["sps_new"]

        if new_sps is not None and current_sps > 0:
            delta = new_sps - current_sps
            pct   = delta / current_sps * 100
        else:
            delta = None
            pct   = None

        row = {
            "model":           model,
            "sps_current":     round(current_sps, 4),
            "sps_with_change": round(new_sps, 4) if new_sps is not None else "",
            "delta":           round(delta, 4)    if delta is not None   else "",
            "pct_change":      round(pct, 2)      if pct is not None     else "",
        }

        (tp_rows if "_tp" in model else non_tp_rows).append(row)

    print(f"  Matched {matched} model(s) between baseline and log data.")

    def sort_key(row):
        # Sort ascending by pct_change (regressions at top); no SPS → sort last
        return row["pct_change"] if row["pct_change"] != "" else float("inf")

    tp_rows.sort(key=sort_key)
    non_tp_rows.sort(key=sort_key)
    return tp_rows, non_tp_rows


def write_csv(rows: list[dict], path: str) -> None:
    fields = ["model", "sps_current", "sps_with_change", "delta", "pct_change"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_summary(tp_rows: list[dict], non_tp_rows: list[dict]) -> None:
    all_rows = tp_rows + non_tp_rows
    regressions  = [r for r in all_rows if r["delta"] != "" and r["delta"] < 0]
    improvements = [r for r in all_rows if r["delta"] != "" and r["delta"] > 0]

    if regressions:
        print(f"\n⚠️  SPS regressions ({len(regressions)}):")
        for r in sorted(regressions, key=lambda x: x["pct_change"]):
            print(f"  {r['model']:<45} {r['pct_change']:+6.1f}%  ({r['sps_current']} → {r['sps_with_change']} SPS)")

    if improvements:
        print(f"\n🚀 SPS improvements ({len(improvements)}):")
        for r in sorted(improvements, key=lambda x: x["pct_change"], reverse=True)[:10]:
            print(f"  {r['model']:<45} {r['pct_change']:+6.1f}%  ({r['sps_current']} → {r['sps_with_change']} SPS)")

    if not regressions:
        print("\n✅ No regressions detected.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SPS from GitHub run logs against latest_perf.csv"
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--run-id",   help="GitHub Actions run ID (downloads logs)")
    src.add_argument("--logs-zip", help="Pre-downloaded logs archive ZIP path")
    src.add_argument("--logs-dir", help="Pre-extracted logs directory path")

    parser.add_argument("--baseline",      default="latest_perf.csv")
    parser.add_argument("--repo",          default="tenstorrent/tt-xla")
    parser.add_argument("--output-prefix", default=None,
                        help="Output CSV filename prefix (default: perf_comparison_<run-id>)")
    args = parser.parse_args()

    if not (args.run_id or args.logs_zip or args.logs_dir):
        parser.error("Provide one of --run-id, --logs-zip, or --logs-dir")

    prefix = args.output_prefix or f"perf_comparison_{args.run_id or 'run'}"

    # 1. Resolve logs directory
    if args.logs_dir:
        logs_dir = args.logs_dir
        print(f"Using existing logs directory: {logs_dir}")
    else:
        if args.logs_zip:
            zip_path = args.logs_zip
        else:
            zip_path = f"/tmp/logs_{args.run_id}.zip"
            download_logs_zip(args.run_id, args.repo, zip_path)

        logs_dir = zip_path.replace(".zip", "_extracted")
        print(f"Extracting {zip_path} → {logs_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(logs_dir)

    # 2. Parse logs
    print(f"\nParsing perf logs in {logs_dir} ...")
    log_results = load_log_results(logs_dir)
    print(f"  {len(log_results)} models parsed from logs.")

    # 3. Load baseline
    print(f"\nLoading baseline from {args.baseline} ...")
    baseline = load_baseline(args.baseline)
    print(f"  {len(baseline)} models in baseline.")

    # 4. Build comparison
    print("\nBuilding comparison ...")
    tp_rows, non_tp_rows = build_rows(baseline, log_results)

    # 5. Write CSVs (skip if empty)
    tp_path     = f"{prefix}_tp.csv"
    non_tp_path = f"{prefix}_non_tp.csv"
    if tp_rows:
        write_csv(tp_rows, tp_path)
        print(f"\nTP CSV    ({len(tp_rows):>3} rows): {tp_path}")
    if non_tp_rows:
        write_csv(non_tp_rows, non_tp_path)
        print(f"Non-TP CSV({len(non_tp_rows):>3} rows): {non_tp_path}")

    # 6. Summary
    print_summary(tp_rows, non_tp_rows)


if __name__ == "__main__":
    main()
