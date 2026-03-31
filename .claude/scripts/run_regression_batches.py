#!/usr/bin/env python3
"""
Run find-regression-boundaries in parallel batches using separate claude processes.

Problems this solves vs running the skill directly on a large failures file:
  1. Output token limit: each claude session only outputs ~10 tests worth of JSON
  2. Parallelism: claude internally caps subagents at ~10 concurrent; this script
     spawns true OS-level parallel processes (one per batch of 10 tests)
  3. Log download race: logs are downloaded ONCE here before any agents start,
     so agents never touch the download logic

Flow:
  1. Read failures JSON
  2. Fetch list of 10 previous CI runs of the same workflow
  3. For each run: if cached check for stub dirs and repair; if not cached download
     all job logs individually via gh api .../jobs/{job_id}/logs  (no ZIP)
  4. Split failed_tests into batches of --batch-size (default 10)
  5. Spawn one `claude -p /find-regression-boundaries` per batch, all in parallel
  6. Wait for all to finish; retry failed batches up to --max-retries times
  7. Merge all per-batch reports into final report (pure Python)
  8. Verify every test from failed_tests has an entry — error if any are missing

Report naming convention (matches the skill):
  Input  batch_0_23375485557_failures.json
  Output regression_report_batch_0_23375485557.json

Usage:
    python3 .claude/scripts/run_regression_batches.py bisection/run_23375485557_failures.json
    python3 .claude/scripts/run_regression_batches.py bisection/run_23375485557_failures.json --batch-size 5
    python3 .claude/scripts/run_regression_batches.py bisection/run_23375485557_failures.json --no-cleanup
"""

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Log pre-download helpers
# ---------------------------------------------------------------------------

def gh_api(endpoint: str, jq: str | None = None, retries: int = 3, retry_delay: float = 5.0) -> str:
    cmd = ["gh", "api", endpoint]
    if jq:
        cmd += ["--jq", jq]
    last_err = ""
    for attempt in range(retries):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        last_err = result.stderr.strip()
        if any(code in last_err for code in ("HTTP 502", "HTTP 503", "HTTP 504", "HTTP 429")):
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
        break
    raise RuntimeError(f"gh api {endpoint} failed:\n{last_err}")


def fetch_run_list(github_repo: str, workflow_id: int, starting_run_id: int) -> list[dict]:
    """Return up to 10 completed runs ordered newest->oldest, starting before starting_run_id."""
    meta_raw = gh_api(
        f"repos/{github_repo}/actions/runs/{starting_run_id}",
        jq='{id: .id, head_sha: .head_sha, created_at: .created_at, conclusion: .conclusion}'
    )
    starting_run = json.loads(meta_raw)
    created_at = starting_run["created_at"]

    collected: list[dict] = []
    page = 1
    while len(collected) < 10:
        raw = gh_api(
            f"repos/{github_repo}/actions/workflows/{workflow_id}/runs"
            f"?per_page=50&status=completed&created=%3C%3D{created_at}&page={page}",
            jq='.workflow_runs[] | {id: .id, head_sha: .head_sha, created_at: .created_at, conclusion: .conclusion}'
        )
        page_runs = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    page_runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not page_runs:
            break
        for r in page_runs:
            if r.get("conclusion") in ("cancelled", "skipped"):
                continue
            collected.append(r)
            if len(collected) >= 10:
                break
        page += 1

    return collected


def load_index(index_path: Path) -> dict:
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return {}


def save_index(index_path: Path, index: dict):
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def fetch_all_run_jobs(run_id: int, github_repo: str) -> list[dict]:
    """Return all jobs for a run (paginated at 25/page to avoid 502s on large responses)."""
    all_jobs = []
    page = 1
    while True:
        raw = gh_api(
            f"repos/{github_repo}/actions/runs/{run_id}/jobs?per_page=25&page={page}",
            jq='.jobs[] | {id: .id, name: .name, conclusion: .conclusion}'
        )
        page_jobs = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    page_jobs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not page_jobs:
            break
        all_jobs.extend(page_jobs)
        if len(page_jobs) < 25:
            break
        page += 1
    return all_jobs


def _fetch_job_log(job_id: int, job_dir: Path, job_name: str, github_repo: str) -> str:
    """Fetch log for one job into job_dir/1_job.txt. Returns 'fetched'/'cached'/'skipped'/'error'."""
    log_file = job_dir / "1_job.txt"
    if log_file.exists():
        return "cached"

    for attempt in range(3):
        result = subprocess.run(
            ["gh", "api", f"repos/{github_repo}/actions/jobs/{job_id}/logs"],
            capture_output=True,
        )
        if result.returncode == 0:
            if result.stdout:
                log_file.write_bytes(result.stdout)
                return "fetched"
            return "skipped"
        err = result.stderr.decode(errors="replace")
        # Skipped/cancelled jobs genuinely have no logs — not an error
        if "Not Found" in err or "410" in err or "404" in err:
            return "skipped"
        # Retry transient errors (including HTTP/2 stream cancellations)
        if any(code in err for code in ("502", "503", "504", "429", "CANCEL")):
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
                continue
        print(f"    WARNING: job {job_id} ({job_name[:60]}): {err[:120]}")
        return "error"

    return "error"


def _print_job_stats(counts: dict) -> None:
    parts = []
    if counts.get("fetched"):
        parts.append(f"{counts['fetched']} fetched")
    if counts.get("cached"):
        parts.append(f"{counts['cached']} already cached")
    if counts.get("skipped"):
        parts.append(f"{counts['skipped']} skipped (no logs)")
    if counts.get("error"):
        parts.append(f"{counts['error']} errors")
    if parts:
        print(f"    {', '.join(parts)}", flush=True)


def download_run_logs(run_id: int, github_repo: str, logs_dir: Path, index: dict) -> dict:
    """Ensure all job logs for a run are downloaded.

    - Cached run: scan for stub-only dirs (system.txt only, left by old ZIP downloads)
      and repair them by fetching each job log individually.
    - Uncached run: list all jobs via API then fetch each job log individually.

    Job dirs: run_{id}/{job_name}/ containing 1_job.txt.
    Job names: API uses ' / ', dirs use ' _ ' (matching GitHub ZIP convention).
    """
    key = f"run_{run_id}"
    run_dir = logs_dir / f"run_{run_id}"

    if key in index:
        # Already indexed — check for stub-only dirs left by old ZIP-based downloads
        if not run_dir.exists():
            print(f"  run {run_id}: cached (dir missing)")
            return index[key]

        stub_dirs = [
            d for d in run_dir.iterdir()
            if d.is_dir() and [f.name for f in d.iterdir()] == ["system.txt"]
        ]
        if not stub_dirs:
            print(f"  run {run_id}: cached, no stubs — OK")
            return index[key]

        print(f"  run {run_id}: cached, {len(stub_dirs)} stub dirs — repairing...", flush=True)
        all_jobs = fetch_all_run_jobs(run_id, github_repo)
        name_to_id = {j["name"].replace(" / ", " _ "): j["id"] for j in all_jobs}

        counts: dict = {}
        for job_dir in stub_dirs:
            job_name = job_dir.name
            job_id = name_to_id.get(job_name)
            if job_id is None:
                print(f"    WARNING: no job ID for: {job_name[:60]}")
                counts["error"] = counts.get("error", 0) + 1
                continue
            status = _fetch_job_log(job_id, job_dir, job_name, github_repo)
            counts[status] = counts.get(status, 0) + 1
        _print_job_stats(counts)
        return index[key]

    # Not cached — fetch metadata then download every job individually
    print(f"  run {run_id}: fetching metadata...", flush=True)
    meta_raw = gh_api(
        f"repos/{github_repo}/actions/runs/{run_id}",
        jq='{id: .id, workflow_id: .workflow_id, workflow_name: (.name), head_sha: .head_sha, created_at: .created_at}'
    )
    meta = json.loads(meta_raw)

    run_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = fetch_all_run_jobs(run_id, github_repo)
    print(f"  run {run_id}: {len(all_jobs)} jobs — downloading logs...", flush=True)

    counts = {}
    for job in all_jobs:
        job_id = job["id"]
        job_name = job["name"].replace(" / ", " _ ")
        job_dir = run_dir / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        status = _fetch_job_log(job_id, job_dir, job_name, github_repo)
        counts[status] = counts.get(status, 0) + 1
    _print_job_stats(counts)

    entry = {
        "run_id": meta["id"],
        "sha": meta["head_sha"],
        "date": meta["created_at"],
        "workflow_id": meta["workflow_id"],
        "workflow_name": meta["workflow_name"],
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    index[key] = entry
    print(f"  run {run_id}: done ({meta['created_at'][:10]})")
    return entry


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def batch_report_path(bisection_dir: Path, batch_idx: int, run_id: int) -> Path:
    """Return the path where the skill will write the report for this batch.
    Matches the skill's filename derivation:
      batch_<i>_<run_id>_failures.json -> regression_report_batch_<i>_<run_id>.json
    """
    return bisection_dir / f"regression_report_batch_{batch_idx}_{run_id}.json"


def batch_input_path(bisection_dir: Path, batch_idx: int, run_id: int) -> Path:
    return bisection_dir / f"batch_{batch_idx}_{run_id}_failures.json"


def write_batch_file(batch_file: Path, data: dict, batch_tests: list) -> None:
    batch_data = {
        "run_id": data["run_id"],
        "run_date": data.get("run_date", ""),
        "sha": data.get("sha", ""),
        "workflow_id": data["workflow_id"],
        "workflow_name": data.get("workflow_name", ""),
        "github_repo": data.get("github_repo", "tenstorrent/tt-xla"),
        # bisect_repo: dedicated clone where git checkout/bisect/tests run — never the script's own repo
        "bisect_repo": data.get("bisect_repo", ""),
        "failed_tests": batch_tests,
        "timed_out_jobs": [],
    }
    with open(batch_file, "w") as f:
        json.dump(batch_data, f, indent=2)


def run_batch(batch_idx: int, batch_file: Path, log_file: Path, repo_root: Path) -> tuple[int, int, bool]:
    """Spawn one `claude -p /find-regression-boundaries <batch_file>` process."""
    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p",
        f"/find-regression-boundaries {batch_file}",
    ]
    with open(log_file, "w") as lf:
        lf.write(f"=== Batch {batch_idx}: {batch_file.name} ===\n")
        lf.write(f"Command: {' '.join(cmd)}\n\n")
        lf.flush()
        result = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(repo_root),
        )
    return batch_idx, result.returncode, result.returncode == 0


def run_batches_parallel(
    batch_indices: list[int],
    bisection_dir: Path,
    batch_logs_dir: Path,
    run_id: int,
    repo_root: Path,
    attempt: int,
) -> tuple[list[int], list[int]]:
    """Run the given batch indices in parallel. Returns (succeeded, failed) index lists."""
    succeeded = []
    failed = []

    with ThreadPoolExecutor(max_workers=len(batch_indices)) as executor:
        futures = {
            executor.submit(
                run_batch,
                i,
                batch_input_path(bisection_dir, i, run_id),
                batch_logs_dir / f"batch_{i}_{run_id}_attempt{attempt}.log",
                repo_root,
            ): i
            for i in batch_indices
        }
        total = len(batch_indices)
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                batch_idx, rc, ok = future.result()
                done += 1
                status = "OK" if ok else f"FAILED (rc={rc})"
                print(f"  [{done}/{total}] batch {batch_idx:2d}  {status}  log: batch_{batch_idx}_{run_id}_attempt{attempt}.log")
                if ok:
                    if batch_report_path(bisection_dir, batch_idx, run_id).exists():
                        succeeded.append(batch_idx)
                    else:
                        print(f"    WARNING: batch {batch_idx} exited OK but report file missing — treating as failed")
                        failed.append(batch_idx)
                else:
                    failed.append(batch_idx)
            except Exception as e:
                done += 1
                print(f"  [{done}/{total}] batch {i:2d}  EXCEPTION: {e}")
                failed.append(i)

    return succeeded, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run find-regression-boundaries in parallel batches"
    )
    parser.add_argument("failures_json", help="Path to failures JSON from /collect-failures")
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Tests per claude process (default: 10). Number of processes = ceil(total / batch_size).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Max retry attempts for batches that fail or produce no report (default: 2)",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep batch JSON and per-batch report files after completion",
    )
    args = parser.parse_args()

    failures_path = Path(args.failures_json).resolve()
    if not failures_path.exists():
        print(f"ERROR: {failures_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(failures_path) as f:
        data = json.load(f)

    run_id = data["run_id"]
    workflow_id = data["workflow_id"]
    github_repo = data.get("github_repo", "tenstorrent/tt-xla")
    failed_tests = data.get("failed_tests", [])

    repo_root = failures_path.parent.parent
    bisection_dir = repo_root / "bisection"
    logs_dir = bisection_dir / "logs"
    index_path = logs_dir / "index.json"

    # bisect_repo: dedicated tt-xla clone used exclusively for git bisect operations
    # (git checkout, submodule updates, test execution). The current repo (repo_root)
    # is NEVER modified — it only stores logs/results and is the CWD for claude agents.
    bisect_repo = repo_root.parent / "tt-xla_bisect"
    if not bisect_repo.exists():
        print(f"ERROR: bisect repo not found at {bisect_repo}", file=sys.stderr)
        print(f"  Create it with:", file=sys.stderr)
        print(f"    git clone <remote_url> {bisect_repo}", file=sys.stderr)
        print(f"  Then set up its venv the same way as this repo.", file=sys.stderr)
        sys.exit(1)

    # Propagate bisect_repo into the data dict so it ends up in every batch JSON file
    data["bisect_repo"] = str(bisect_repo)

    n_tests = len(failed_tests)
    n_batches = math.ceil(n_tests / args.batch_size)

    print(f"Failures file:  {failures_path.relative_to(repo_root)}")
    print(f"Run ID:         {run_id}")
    print(f"Script repo:    {repo_root}  (never modified — logs/results live here)")
    print(f"Bisect repo:    {bisect_repo}  (git checkout/bisect/tests run here)")
    print(f"Failed tests:   {n_tests}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Processes:      {n_batches}  (one per batch, all in parallel)")
    print(f"Max retries:    {args.max_retries}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: fetch run list
    # ------------------------------------------------------------------
    print("Fetching previous run list...")
    run_list = fetch_run_list(github_repo, workflow_id, run_id)
    print(f"Runs to search ({len(run_list)}, newest -> oldest, prior to run {run_id}):")
    for idx, r in enumerate(run_list):
        print(f"  [{idx}] run {r['id']}  {r['created_at'][:10]}  sha={r['head_sha'][:8]}")
    print()

    # ------------------------------------------------------------------
    # Phase 2: ensure all run logs are downloaded (sequential, no race)
    #   - cached runs: check for stub dirs, repair any found
    #   - uncached runs: fetch all job logs individually via jobs API
    # ------------------------------------------------------------------
    print("Checking / downloading run logs...")
    logs_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(index_path)
    for r in run_list:
        try:
            download_run_logs(r["id"], github_repo, logs_dir, index)
        except RuntimeError as e:
            print(f"  WARNING: {e} — this run will be skipped by agents")
        save_index(index_path, index)
    print()

    # ------------------------------------------------------------------
    # Phase 3: ensure GitHub SSH host key is known (avoids interactive
    # prompts when 8 agents all connect simultaneously)
    # ------------------------------------------------------------------
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    known_hosts.parent.mkdir(exist_ok=True)
    existing = known_hosts.read_text() if known_hosts.exists() else ""
    if "github.com" not in existing:
        print("Adding GitHub to SSH known_hosts...")
        result = subprocess.run(
            ["ssh-keyscan", "github.com"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            with open(known_hosts, "a") as f:
                f.write(result.stdout)
            print("  done")
        else:
            print("  WARNING: ssh-keyscan failed — agents may hang on SSH prompt")
    print()

    # ------------------------------------------------------------------
    # Phase 4: write batch JSON files
    # ------------------------------------------------------------------
    batch_logs_dir = bisection_dir / "batch_logs"
    batch_logs_dir.mkdir(exist_ok=True)

    print(f"Writing {n_batches} batch files:")
    for i in range(n_batches):
        batch_tests = failed_tests[i * args.batch_size : (i + 1) * args.batch_size]
        write_batch_file(batch_input_path(bisection_dir, i, run_id), data, batch_tests)
        print(f"  batch {i:2d}: {len(batch_tests):2d} tests -> batch_{i}_{run_id}_failures.json")
    print()

    # ------------------------------------------------------------------
    # Phase 5: run all batches, with retries for failures
    # ------------------------------------------------------------------
    pending = list(range(n_batches))
    all_succeeded = []
    start_time = time.time()

    for attempt in range(1, args.max_retries + 2):  # +2: initial run + max_retries
        if not pending:
            break
        label = "initial run" if attempt == 1 else f"retry {attempt - 1}/{args.max_retries}"
        print(f"Launching {len(pending)} claude processes in parallel ({label})...")

        succeeded, failed = run_batches_parallel(
            pending, bisection_dir, batch_logs_dir, run_id, repo_root, attempt
        )
        all_succeeded.extend(succeeded)
        pending = failed

        if pending and attempt <= args.max_retries:
            print(f"\n{len(pending)} batch(es) failed: {pending} — retrying...")
        elif pending:
            print(f"\nERROR: {len(pending)} batch(es) still failing after {args.max_retries} retries: {pending}")
            print(f"  Check logs in bisection/batch_logs/")

    total_elapsed = time.time() - start_time
    print(f"\nAll batches done in {total_elapsed:.0f}s")
    print()

    # ------------------------------------------------------------------
    # Phase 6: merge per-batch reports (pure Python)
    # ------------------------------------------------------------------
    print("Merging results...")
    all_results = []

    for i in range(n_batches):
        report_file = batch_report_path(bisection_dir, i, run_id)
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)
            batch_results = report.get("results", [])
            all_results.extend(batch_results)
            print(f"  batch {i:2d}: {len(batch_results)} results")
        else:
            print(f"  batch {i:2d}: WARNING — no report file ({report_file.name})")

    # ------------------------------------------------------------------
    # Phase 7: verify every test is present in results
    # ------------------------------------------------------------------
    result_keys = {(r["test_id"], r["machine_type"]) for r in all_results}
    input_keys = {(t["test_id"], t["machine_type"]) for t in failed_tests}
    missing_tests = input_keys - result_keys

    if missing_tests:
        print(f"\nERROR: {len(missing_tests)} test(s) missing from final report:")
        for test_id, machine_type in sorted(missing_tests):
            print(f"  {test_id}  [{machine_type}]")
        print("  Re-run the script or check the batch logs for the affected batches.")
    else:
        print(f"\nAll {n_tests} tests accounted for.")

    # ------------------------------------------------------------------
    # Phase 8: write merged report
    # ------------------------------------------------------------------
    boundaries_found = sum(1 for r in all_results if r.get("boundary_found"))
    merged_report = {
        "source_run_id": run_id,
        "source_run_date": data.get("run_date", ""),
        "workflow_name": data.get("workflow_name", ""),
        "github_repo": github_repo,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tests": len(all_results),
        "boundaries_found": boundaries_found,
        "boundaries_not_found": len(all_results) - boundaries_found,
        "results": all_results,
    }

    output_file = bisection_dir / f"regression_report_{run_id}.json"
    with open(output_file, "w") as f:
        json.dump(merged_report, f, indent=2)

    print(f"Merged report:  {output_file.relative_to(repo_root)}")
    print(f"  Total tests:        {len(all_results)}")
    print(f"  Boundaries found:   {boundaries_found}")
    print(f"  Boundaries missing: {len(all_results) - boundaries_found}")
    if missing_tests:
        print(f"  Tests absent:       {len(missing_tests)}  <- ERROR")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if not args.no_cleanup:
        for i in range(n_batches):
            batch_input_path(bisection_dir, i, run_id).unlink(missing_ok=True)
            batch_report_path(bisection_dir, i, run_id).unlink(missing_ok=True)
        print("\nCleaned up batch files and per-batch reports.")

    if missing_tests:
        sys.exit(1)


if __name__ == "__main__":
    main()
