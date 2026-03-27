#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Launch a Claude instance for each model in top_models.json.

Each worker gets its own git worktree and branch so parallel Claude
instances never run git commands against the same working directory.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_DIR / "scripts" / "logs"
WORKTREE_DIR = REPO_DIR / ".worktrees"


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

    return model_id, result.returncode


def run_model_in_worktree(index: int, model_id: str, base_branch: str):
    """Ensure a worktree exists for this worker, then run the model."""
    branch = f"{base_branch}-{index}"
    worktree_path = WORKTREE_DIR / f"worker-{index}"

    try:
        ensure_worktree(branch, worktree_path)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create worktree for {model_id}: {e.stderr}", file=sys.stderr)
        return model_id, 1

    return run_model(model_id, worktree_path)


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
        "--workers", type=int, default=16, help="Max concurrent Claude instances"
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

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_model_in_worktree, i, mid, base_branch): mid
            for i, mid in enumerate(model_ids)
        }

        for i, future in enumerate(as_completed(futures), 1):
            model_id, rc = future.result()
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"[{i}/{len(model_ids)}] {model_id}: {status}")

    print("All done.")


if __name__ == "__main__":
    main()
