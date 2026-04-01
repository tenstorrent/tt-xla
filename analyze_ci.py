#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fetch and analyze CI test results for GitHub Actions runs.

Usage:
    python3 analyze_ci.py nightly:23833071855 weekly:23712685414
    python3 analyze_ci.py nightly:https://github.com/tenstorrent/tt-xla/actions/runs/23833071855
    python3 analyze_ci.py 23833071855  # label defaults to run ID

Downloads test job logs and runs parse_ci_tests.py on each.
Handles re-run attempts (picks the latest attempt per job).
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def get_repo():
    """Detect GitHub repo from git remote."""
    r = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
    )
    url = r.stdout.strip().rstrip(".git")
    for sep in ("github.com:", "github.com/"):
        if sep in url:
            return url.split(sep, 1)[1]
    raise RuntimeError(f"Cannot parse repo from remote: {url}")


def parse_run_spec(spec):
    """Parse '[label:]RUN_ID_or_URL' into (label, run_id, repo_or_None).

    Accepted formats:
        nightly:23833071855
        nightly:https://github.com/owner/repo/actions/runs/23833071855
        23833071855
        https://github.com/owner/repo/actions/runs/23833071855
    """
    # Split label from value
    m = re.match(r"^([A-Za-z][\w-]*):(.*)", spec)
    if m:
        label, value = m.group(1), m.group(2)
    else:
        label, value = None, spec

    # Extract repo + run_id from URL, or treat value as bare run ID
    url_m = re.search(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)", value)
    if url_m:
        repo, run_id = url_m.group(1), url_m.group(2)
    else:
        repo, run_id = None, value.strip()

    if label is None:
        label = run_id

    return label, run_id, repo


def fetch_test_jobs(repo, run_id):
    """Fetch test jobs for a run, deduped to latest attempt per job name."""
    r = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/jobs?filter=all&per_page=100",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    jobs = json.loads(r.stdout).get("jobs", [])

    # Keep only actual test runner jobs (not matrix generation, build, notify)
    test_jobs = [j for j in jobs if "/ test " in j["name"]]

    # Deduplicate: latest attempt per job name
    by_name = {}
    for j in test_jobs:
        name = j["name"]
        attempt = j.get("run_attempt", 1)
        if name not in by_name or attempt > by_name[name].get("run_attempt", 1):
            by_name[name] = j

    return list(by_name.values())


def download_log(repo, run_id, job, outdir):
    """Download log for a single job. Returns the output path."""
    job_id = job["id"]
    outfile = outdir / f"{job_id}.log"

    r = subprocess.run(
        [
            "gh",
            "run",
            "view",
            str(run_id),
            "--repo",
            repo,
            "--log",
            "--job",
            str(job_id),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if r.returncode == 0 and r.stdout.strip():
        outfile.write_text(r.stdout)
        return outfile

    # Fallback: direct job logs API
    r2 = subprocess.run(
        ["gh", "api", f"repos/{repo}/actions/jobs/{job_id}/logs"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if r2.returncode == 0 and r2.stdout.strip():
        outfile.write_text(r2.stdout)
    else:
        outfile.write_text(f"log not found for job {job_id}\n{r.stderr}\n{r2.stderr}")

    return outfile


def analyze(outdir):
    """Run parse_ci_tests.py on a log directory."""
    script = Path(__file__).parent / "parse_ci_tests.py"
    r = subprocess.run(
        [sys.executable, str(script), str(outdir)],
        capture_output=True,
        text=True,
    )
    return r.stdout + r.stderr


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="+", metavar="[label:]RUN_ID_or_URL")
    p.add_argument(
        "--repo", default=None, help="GitHub repo (auto-detected if omitted)"
    )
    p.add_argument("--outdir", default="ci-logs", help="Base output directory")
    p.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Re-analyze existing logs without downloading",
    )
    args = p.parse_args()

    base = Path(args.outdir)

    # Parse specs and auto-detect repo
    runs = []
    repo = args.repo
    for spec in args.runs:
        label, run_id, spec_repo = parse_run_spec(spec)
        if repo is None and spec_repo:
            repo = spec_repo
        runs.append((label, run_id))

    if repo is None:
        repo = get_repo()

    print(f"Repo: {repo}")

    for label, run_id in runs:
        print(f"\n{'=' * 60}")
        print(f"  {label}  (run {run_id})")
        print(f"{'=' * 60}")

        outdir = base / label

        if not args.skip_download:
            # Fetch job list
            print("Fetching job list...")
            jobs = fetch_test_jobs(repo, run_id)
            print(f"Found {len(jobs)} test jobs")

            reattempts = [j for j in jobs if j.get("run_attempt", 1) > 1]
            if reattempts:
                for j in reattempts:
                    print(f"  re-run (attempt {j['run_attempt']}): {j['name']}")

            # Clear and (re-)create output dir
            outdir.mkdir(parents=True, exist_ok=True)
            for old in outdir.glob("*.log"):
                old.unlink()

            # Download logs in parallel
            print(f"Downloading logs to {outdir}/...")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {
                    pool.submit(download_log, repo, run_id, j, outdir): j for j in jobs
                }
                for i, f in enumerate(as_completed(futs), 1):
                    f.result()  # propagate exceptions
                    sys.stdout.write(f"  {i}/{len(jobs)}\r")
                    sys.stdout.flush()
            print()

        # Analyze
        print(f"\n--- Results: {label} ---\n")
        print(analyze(outdir))


if __name__ == "__main__":
    main()
