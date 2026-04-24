#!/usr/bin/env python3
"""
Batch-download perf report artifacts for a GitHub Actions run and extract metrics.

Usage:
    python fetch_perf_reports.py <RUN_ID> [--output-dir DIR] [--repo OWNER/REPO]

Outputs:
    - One JSON file per perf-report artifact in the output directory
    - summary.json with extracted metrics for all jobs
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def run_gh(args: list[str], parse_json: bool = True):
    """Run a gh CLI command and return parsed output."""
    cmd = ["gh", "api"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None
    if parse_json:
        return json.loads(result.stdout)
    return result.stdout


def get_run_jobs(repo: str, run_id: str) -> list[dict]:
    """Get all jobs for a run, handling pagination."""
    jobs = []
    page = 1
    while True:
        data = run_gh(
            [f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&page={page}"]
        )
        if not data or not data.get("jobs"):
            break
        jobs.extend(data["jobs"])
        if len(data["jobs"]) < 100:
            break
        page += 1
    return jobs


def get_artifacts(repo: str, run_id: str) -> list[dict]:
    """Get all artifacts for a run, handling pagination."""
    artifacts = []
    page = 1
    while True:
        data = run_gh(
            [f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"]
        )
        if not data or not data.get("artifacts"):
            break
        artifacts.extend(data["artifacts"])
        if len(data["artifacts"]) < 100:
            break
        page += 1
    return artifacts


def download_artifact(repo: str, artifact_id: int, output_dir: Path) -> Path | None:
    """Download and extract a single artifact zip."""
    zip_path = output_dir / f"artifact_{artifact_id}.zip"
    extract_dir = output_dir / f"artifact_{artifact_id}"

    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/actions/artifacts/{artifact_id}/zip"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"Failed to download artifact {artifact_id}", file=sys.stderr)
        return None

    zip_path.write_bytes(result.stdout)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            json_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not json_files:
                return None
            extract_dir.mkdir(exist_ok=True)
            for jf in json_files:
                zf.extract(jf, extract_dir)
        return extract_dir
    except zipfile.BadZipFile:
        print(f"Bad zip for artifact {artifact_id}", file=sys.stderr)
        return None
    finally:
        zip_path.unlink(missing_ok=True)


def extract_metrics(report: dict) -> dict:
    """Extract key metrics from a perf report JSON."""
    measurements = {m["measurement_name"]: m["value"] for m in report.get("measurements", [])}

    total_samples = measurements.get("total_samples", 0)
    total_time = measurements.get("total_time", 0)
    samples_per_sec = total_samples / total_time if total_time > 0 else None

    return {
        "model": report.get("model", "unknown"),
        "display_name": report.get("config", {}).get("display_name", "unknown"),
        "device_type": report.get("device_info", {}).get("device_type", "unknown"),
        "device_count": report.get("device_info", {}).get("device_count", 1),
        "arch": report.get("device_info", {}).get("arch", "unknown"),
        "batch_size": report.get("batch_size"),
        "input_sequence_length": report.get("input_sequence_length"),
        "samples_per_sec": round(samples_per_sec, 2) if samples_per_sec else None,
        "ttft_ms": round(measurements["ttft"], 2) if "ttft" in measurements else None,
        "device_fw_duration_s": (
            round(measurements["device_fw_duration"], 4)
            if "device_fw_duration" in measurements
            else None
        ),
        "device_kernel_duration_s": (
            round(measurements["device_kernel_duration"], 4)
            if "device_kernel_duration" in measurements
            else None
        ),
        "total_samples": total_samples,
        "total_time_s": round(total_time, 4) if total_time else None,
        "job_id_url": report.get("config", {}).get("job_id_url"),
        "num_graphs": report.get("config", {}).get("ttnn_num_graphs"),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch and process perf report artifacts")
    parser.add_argument("run_id", help="GitHub Actions run ID")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs (default: /tmp/perf_reports_<RUN_ID>)",
    )
    parser.add_argument(
        "--repo",
        default="tenstorrent/tt-xla",
        help="GitHub repo (default: tenstorrent/tt-xla)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"/tmp/perf_reports_{args.run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching jobs for run {args.run_id}...", file=sys.stderr)
    jobs = get_run_jobs(args.repo, args.run_id)

    # Identify perf benchmark jobs
    perf_jobs = {}
    for job in jobs:
        if "perf-benchmark" in job["name"] and "perf " in job["name"]:
            perf_jobs[str(job["id"])] = {
                "name": job["name"],
                "conclusion": job["conclusion"],
                "url": job["url"],
            }

    print(f"Found {len(perf_jobs)} perf benchmark jobs", file=sys.stderr)

    print("Fetching artifacts...", file=sys.stderr)
    artifacts = get_artifacts(args.repo, args.run_id)

    # Match perf-report artifacts to jobs
    perf_artifacts = []
    for art in artifacts:
        if art["name"].startswith("perf-reports-"):
            if art.get("expired"):
                print(f"Artifact expired: {art['name']}", file=sys.stderr)
                continue
            perf_artifacts.append(art)

    print(f"Found {len(perf_artifacts)} perf report artifacts", file=sys.stderr)

    # Download and process each artifact
    all_metrics = []
    for art in perf_artifacts:
        print(f"  Downloading {art['name']}...", file=sys.stderr)
        extract_dir = download_artifact(args.repo, art["id"], output_dir)
        if not extract_dir:
            continue

        # Find and process JSON files
        for json_file in extract_dir.rglob("*.json"):
            try:
                report = json.loads(json_file.read_text())
                # Copy report to output dir with a meaningful name
                display = report.get("config", {}).get("display_name", "unknown")
                dest = output_dir / f"{display}.json"
                json_file.rename(dest) if not dest.exists() else None
                metrics = extract_metrics(report)
                metrics["artifact_name"] = art["name"]
                all_metrics.append(metrics)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Error processing {json_file}: {e}", file=sys.stderr)

    # Sort by device type then model name
    all_metrics.sort(key=lambda m: (m["device_type"], m["display_name"]))

    # Write summary
    summary = {
        "run_id": args.run_id,
        "total_perf_jobs": len(perf_jobs),
        "successful_reports": len(all_metrics),
        "failed_jobs": [
            {"name": j["name"], "url": j["url"]}
            for j in perf_jobs.values()
            if j["conclusion"] == "failure"
        ],
        "metrics": all_metrics,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}", file=sys.stderr)

    # Also print summary to stdout for piping
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
