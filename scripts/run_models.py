#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Launch a Claude instance for each model in top_models.json."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_DIR / "scripts" / "logs"


def run_model(model_id: str):
    """Run claude on a single model, returning (model_id, returncode)."""
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
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    return model_id, result.returncode


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

    with open(args.json_file) as f:
        models = json.load(f)

    model_ids = [m["id"] for m in models]
    if args.limit > 0:
        model_ids = model_ids[: args.limit]

    if args.start > 0:
        model_ids = model_ids[args.start :]

    print(f"Processing {len(model_ids)} models with {args.workers} workers")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_model, mid): mid for mid in model_ids}

        for i, future in enumerate(as_completed(futures), 1):
            model_id, rc = future.result()
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"[{i}/{len(model_ids)}] {model_id}: {status}")

    print("All done.")


if __name__ == "__main__":
    main()
