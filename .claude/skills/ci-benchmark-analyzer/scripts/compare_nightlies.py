#!/usr/bin/env python3
"""
Compare perf metrics between two nightly CI runs and produce a structured diff.

Usage:
    python compare_nightlies.py <CURRENT_RUN_ID> <PREVIOUS_RUN_ID> \
        [--repo OWNER/REPO] [--output-dir DIR] [--threshold 10]

Outputs summary.json with per-model diffs categorized as regression/improvement/stable.

The comparison is based on samples/sec (total_samples / total_time).
Perf diff convention:
  - POSITIVE % = model got FASTER (improvement)
  - NEGATIVE % = model got SLOWER (regression)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def run_gh(args: list[str]):
    """Run a gh API command and return parsed JSON."""
    cmd = ["gh", "api"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None
    return json.loads(result.stdout)


def get_perf_artifacts(repo: str, run_id: str) -> list[dict]:
    """Get perf-report artifacts for a run."""
    artifacts = []
    page = 1
    while True:
        data = run_gh(
            [f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"]
        )
        if not data or not data.get("artifacts"):
            break
        for a in data["artifacts"]:
            if a["name"].startswith("perf-reports-") and not a.get("expired"):
                artifacts.append(a)
        if len(data["artifacts"]) < 100:
            break
        page += 1
    return artifacts


def download_and_parse_report(repo: str, artifact_id: int, tmp_dir: Path) -> dict | None:
    """Download a perf-report artifact and parse its JSON."""
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/actions/artifacts/{artifact_id}/zip"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None

    zip_path = tmp_dir / f"art_{artifact_id}.zip"
    zip_path.write_bytes(result.stdout)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".json") and "perf_report" in name:
                    with zf.open(name) as f:
                        return json.loads(f.read())
    except (zipfile.BadZipFile, json.JSONDecodeError):
        pass
    finally:
        zip_path.unlink(missing_ok=True)
    return None


def extract_model_metrics(report: dict) -> dict:
    """Extract key metrics from a perf report."""
    measurements = {m["measurement_name"]: m["value"] for m in report.get("measurements", [])}
    total_samples = measurements.get("total_samples", 0)
    total_time = measurements.get("total_time", 0)

    return {
        "display_name": report.get("config", {}).get("display_name", "unknown"),
        "device_type": report.get("device_info", {}).get("device_type", "unknown"),
        "samples_per_sec": total_samples / total_time if total_time > 0 else 0,
        "ttft_ms": measurements.get("ttft"),
        "device_fw_duration_s": measurements.get("device_fw_duration"),
        "total_samples": total_samples,
        "total_time_s": total_time,
        "job_id_url": report.get("config", {}).get("job_id_url"),
    }


def compute_diff(current: float, previous: float) -> float | None:
    """Compute percent change. Positive = improvement (faster), negative = regression (slower)."""
    if previous <= 0:
        return None
    return round(((current - previous) / previous) * 100, 2)


def main():
    parser = argparse.ArgumentParser(description="Compare perf between two CI runs")
    parser.add_argument("current_run_id", help="Current (newer) run ID")
    parser.add_argument("previous_run_id", help="Previous (older) run ID")
    parser.add_argument("--repo", default="tenstorrent/tt-xla")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Percent threshold for flagging regressions/improvements (default: 10)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"/tmp/perf_compare_{args.current_run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp())

    # Fetch artifacts for both runs
    print(f"Fetching artifacts for current run {args.current_run_id}...", file=sys.stderr)
    current_artifacts = get_perf_artifacts(args.repo, args.current_run_id)
    print(f"  Found {len(current_artifacts)} perf report artifacts", file=sys.stderr)

    print(f"Fetching artifacts for previous run {args.previous_run_id}...", file=sys.stderr)
    previous_artifacts = get_perf_artifacts(args.repo, args.previous_run_id)
    print(f"  Found {len(previous_artifacts)} perf report artifacts", file=sys.stderr)

    # Download and parse all reports
    print("Downloading current run reports...", file=sys.stderr)
    current_models = {}
    for art in current_artifacts:
        report = download_and_parse_report(args.repo, art["id"], tmp_dir)
        if report:
            metrics = extract_model_metrics(report)
            key = f"{metrics['display_name']}_{metrics['device_type']}"
            current_models[key] = metrics

    print("Downloading previous run reports...", file=sys.stderr)
    previous_models = {}
    for art in previous_artifacts:
        report = download_and_parse_report(args.repo, art["id"], tmp_dir)
        if report:
            metrics = extract_model_metrics(report)
            key = f"{metrics['display_name']}_{metrics['device_type']}"
            previous_models[key] = metrics

    print(
        f"Parsed {len(current_models)} current, {len(previous_models)} previous models",
        file=sys.stderr,
    )

    # Compute diffs
    comparisons = []
    for key, curr in sorted(current_models.items()):
        prev = previous_models.get(key)
        if not prev:
            comparisons.append(
                {
                    **curr,
                    "previous_samples_per_sec": None,
                    "previous_ttft_ms": None,
                    "diff_percent": None,
                    "category": "new",
                    "diff_ttft_percent": None,
                }
            )
            continue

        sps_diff = compute_diff(curr["samples_per_sec"], prev["samples_per_sec"])
        ttft_diff = None
        if curr.get("ttft_ms") is not None and prev.get("ttft_ms") is not None and prev["ttft_ms"] > 0:
            # For TTFT, lower is better, so invert: negative change in TTFT = improvement
            ttft_diff = round(((prev["ttft_ms"] - curr["ttft_ms"]) / prev["ttft_ms"]) * 100, 2)

        if sps_diff is None:
            category = "unknown"
        elif sps_diff <= -args.threshold:
            category = "regression"
        elif sps_diff >= args.threshold:
            category = "improvement"
        else:
            category = "stable"

        comparisons.append(
            {
                **curr,
                "previous_samples_per_sec": round(prev["samples_per_sec"], 2),
                "previous_ttft_ms": round(prev["ttft_ms"], 2) if prev.get("ttft_ms") is not None else None,
                "diff_percent": sps_diff,
                "category": category,
                "diff_ttft_percent": ttft_diff,
            }
        )

    # Also find models that were in previous but missing from current (disappeared)
    for key, prev in sorted(previous_models.items()):
        if key not in current_models:
            comparisons.append(
                {
                    "display_name": prev["display_name"],
                    "device_type": prev["device_type"],
                    "samples_per_sec": None,
                    "previous_samples_per_sec": round(prev["samples_per_sec"], 2),
                    "previous_ttft_ms": round(prev["ttft_ms"], 2) if prev.get("ttft_ms") is not None else None,
                    "diff_percent": None,
                    "category": "missing",
                    "ttft_ms": None,
                    "device_fw_duration_s": None,
                    "total_samples": None,
                    "total_time_s": None,
                    "job_id_url": None,
                    "diff_ttft_percent": None,
                }
            )

    # Categorize
    regressions = [c for c in comparisons if c["category"] == "regression"]
    improvements = [c for c in comparisons if c["category"] == "improvement"]
    stable = [c for c in comparisons if c["category"] == "stable"]
    new_models = [c for c in comparisons if c["category"] == "new"]
    missing = [c for c in comparisons if c["category"] == "missing"]

    summary = {
        "current_run_id": args.current_run_id,
        "previous_run_id": args.previous_run_id,
        "threshold_percent": args.threshold,
        "total_models_compared": len([c for c in comparisons if c["category"] not in ("new", "missing")]),
        "regressions": len(regressions),
        "improvements": len(improvements),
        "stable": len(stable),
        "new_models": len(new_models),
        "missing_models": len(missing),
        "models": {
            "regressions": sorted(regressions, key=lambda x: x["diff_percent"] or 0),
            "improvements": sorted(
                improvements, key=lambda x: x["diff_percent"] or 0, reverse=True
            ),
            "stable": sorted(stable, key=lambda x: x["display_name"]),
            "new": new_models,
            "missing": missing,
        },
        "all_comparisons": sorted(comparisons, key=lambda x: (x["device_type"], x["display_name"])),
    }

    summary_path = output_dir / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nComparison written to {summary_path}", file=sys.stderr)

    # Print summary to stdout
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
