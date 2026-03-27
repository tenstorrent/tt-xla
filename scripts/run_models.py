#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Launch a Claude instance for each model in top_models.json.

Each worker gets its own git worktree and branch so parallel Claude
instances never run git commands against the same working directory.
"""

import argparse
import datetime
import json
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_DIR / "scripts" / "logs"
WORKTREE_DIR = REPO_DIR / ".worktrees"


class TeeStream:
    """Write to both a file and the original stream."""

    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


def get_current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def ensure_worktree(branch: str, worktree_path: Path):
    """Reuse an existing worktree or create a new one."""
    if worktree_path.exists():
        return
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(worktree_path), "HEAD"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True,
    )


def run_model(model_id: str, worktree_path: Path):
    """Run claude on a single model inside a dedicated worktree."""
    log_file = LOG_DIR / f"{model_id.replace('/', '_')}.log"

    with open(log_file, "w") as log:
        result = subprocess.run(
            [
                "claude",
                "-p",
                f"/port-huggingface-model {model_id}",
                "--allowedTools",
                "Edit,Write,Read,Glob,Grep,Bash,Skill,Agent",
            ],
            cwd=worktree_path,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    reason = ""
    if result.returncode != 0:
        try:
            last_line = log_file.read_text().strip().rsplit("\n", 1)[-1]
            reason = last_line[:200]
        except OSError:
            reason = "could not read log"

    return model_id, result.returncode, reason


def run_worker(
    worker_index: int,
    model_ids,
    base_branch: str,
    result_queue: multiprocessing.Queue,
):
    """Run all assigned models sequentially in this worker's worktree."""
    branch = f"{base_branch}-{worker_index}"
    worktree_path = WORKTREE_DIR / f"worker-{worker_index}"

    try:
        ensure_worktree(branch, worktree_path)
    except subprocess.CalledProcessError as e:
        for mid in model_ids:
            result_queue.put(
                (worker_index, mid, 1, f"worktree creation failed: {e.stderr.strip()}")
            )
        return

    for mid in model_ids:
        _, rc, reason = run_model(mid, worktree_path)
        result_queue.put((worker_index, mid, rc, reason))


def main():
    parser = argparse.ArgumentParser(description="Run Claude on top HuggingFace models")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset models to process (default = 0)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max models to process (0 = all)"
    )
    parser.add_argument(
        "--workers", type=int, default=25, help="Max concurrent Claude instances"
    )
    parser.add_argument(
        "json_file",
        type=str,
        default=None,
        help="json file path",
    )
    args = parser.parse_args()

    if not Path(args.json_file).exists():
        print(f"Error: {args.json_file} not found", file=sys.stderr)
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WORKTREE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log = LOG_DIR / f"run_models_{timestamp}.log"
    log_fh = open(main_log, "w")
    sys.stdout = TeeStream(sys.__stdout__, log_fh)
    sys.stderr = TeeStream(sys.__stderr__, log_fh)
    print(f"Logging to {main_log}")

    base_branch = get_current_branch()

    with open(args.json_file) as f:
        models = json.load(f)

    model_ids = [m["id"] for m in models]

    if args.start > 0:
        model_ids = model_ids[args.start :]

    if args.limit > 0:
        model_ids = model_ids[: args.limit]

    print(f"Processing {len(model_ids)} models with {args.workers} workers")
    print(f"Base branch: {base_branch}")

    num_workers = min(args.workers, len(model_ids))
    chunks = [[] for _ in range(num_workers)]
    for i, mid in enumerate(model_ids):
        chunks[i % num_workers].append(mid)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(run_worker, worker_idx, chunk, base_branch, result_queue)
            for worker_idx, chunk in enumerate(chunks)
        ]

        print("Workers started...")

        completed = 0
        while completed < len(model_ids):
            worker_index, model_id, rc, reason = result_queue.get()
            completed += 1
            status = "OK" if rc == 0 else f"FAILED (rc={rc}): {reason}"
            print(
                f"[{completed}/{len(model_ids)}] worker-{worker_index} {model_id}: {status}"
            )

        for future in futures:
            future.result()

    print("All done.")
    log_fh.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
