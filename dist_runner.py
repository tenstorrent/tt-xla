#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import getpass
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Run git pull, submodules, venv, tt-smi, pytest on a remote machine over SSH, tee output locally."
    )
    p.add_argument("host", help="Remote host (e.g. gpu01 or gpu01.example.com)")
    p.add_argument("--port", type=int, default=47609, help="SSH port (default: 22)")
    p.add_argument(
        "--remote-dir",
        default="/localdev/ndrakulic/tt-xla",
        help="Remote working directory (default: /localdev/ndrakulic/tt-xla)",
    )
    p.add_argument(
        "--venv-activate",
        default="venv/activate",
        help="Path to venv activate script (default: venv/bin/activate)",
    )
    p.add_argument(
        "--pytest-target",
        default="",
        help="Pytest target, e.g. \"path/to/test_file.py::TestClass::test_name\" or 'path[some_test]'",
    )
    p.add_argument(
        "--log", help="Local log file path (default: ./remote_run_YYYYmmdd_HHMMSS.log)"
    )
    args = p.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    host = args.host
    pytest_target = args.pytest_target
    log_path = (
        Path(args.log)
        if args.log
        else Path(f"{host}__{pytest_target.replace('/', '_')}__{timestamp}.log")
    )
    log_path = "logs" / log_path
    # Build the remote command; run in bash with 'set -euo pipefail' so failures stop the chain.
    remote_dir = args.remote_dir
    venv_act = args.venv_activate
    pytest_target = args.pytest_target
    pytest_cmd = f"pytest tests/runner/test_models.py::test_all_models[{pytest_target}]"

    remote_cmd = (
        "set -euo pipefail; "
        "source /home/ndrakulic/.bashrc; "
        f"cd {remote_dir}; "
        "git pull; "
        "git submodule update --init --recursive; "
        f"source {venv_act}; "
        "tt-smi -r; "
        f"{pytest_cmd} "
    )

    ssh_cmd = ["ssh", "-tt", "-p", str(args.port)]
    ssh_cmd += [f"{args.host}", "bash", "-lc", remote_cmd]

    print("==> Executing on remote host:", *ssh_cmd[:4], "...", file=sys.stderr)
    print(f"==> Logging to: {log_path}", file=sys.stderr)

    with log_path.open("w", encoding="utf-8") as logf:
        # Stream stdout+stderr and tee to log
        proc = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
            logf.flush()
        proc.wait()

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
