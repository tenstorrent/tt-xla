#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def make_remote_script(remote_dir, venv_act, pytest_cmd):
    long_cmd = (
        # "set -euo pipefail; "
        "source /home/ndrakulic/.bashrc; "
        f"cd {remote_dir}; "
        "git pull; "
        "git submodule update --init --recursive; "
        f"source {venv_act}; "
        "tt-smi -r; "
        f"{pytest_cmd}"
    )

    # NOTE: no extra quoting around long_cmd here, we let tmux send-keys the raw string
    return f"""tmux has-session -t tests || tmux new-session -d -s tests
tmux send-keys -t tests "{long_cmd}" C-m
"""


def main():
    p = argparse.ArgumentParser(
        description="Launch pytest remotely in tmux 'tests' via SSH (non-blocking)."
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
        help='Pytest param for test_all_models[...] e.g. "model_name" or "model[param]"',
    )
    p.add_argument(
        "--log",
        help="Local log file path (default: ./logs/<host>__<target>__YYYYmmdd_HHMMSS.log)",
    )
    args = p.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pytest_target = args.pytest_target
    pytest_report_file = f"pytest_reports/{args.host}_{timestamp}.xml"
    # what we'll actually run in tmux
    pytest_cmd = f"pytest --junitxml={pytest_report_file} tests/runner/test_models.py::test_all_models[{pytest_target}]"

    # local log path (we just record what we triggered remotely)
    suggested_name = f"{args.host}__{pytest_target.replace('/', '_')}__{timestamp}.log"
    log_path = Path(args.log) if args.log else Path("logs") / suggested_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # build the tmux remote script
    remote_script = make_remote_script(args.remote_dir, args.venv_activate, pytest_cmd)

    ssh_cmd = [
        "ssh",
        "-tt",
        "-p",
        str(args.port),
        args.host,
        "bash",
        "-lc",
        "cat | bash",
    ]

    # log what we are about to send
    print("== remote script =================", file=sys.stderr)
    print(remote_script, file=sys.stderr)
    print("=================================", file=sys.stderr)

    proc = subprocess.Popen(
        ssh_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdin is not None
    assert proc.stdout is not None

    # send the script body over stdin to remote "cat | bash"
    proc.stdin.write(remote_script)
    proc.stdin.close()

    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
