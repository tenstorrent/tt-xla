#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time
import shlex

from sample_execution_list import execution_list

PORT = 47609
REMOTE_DIR = "/localdev/ndrakulic/tt-xla"

def make_remote_script(pytest_cmd, remote_dir=REMOTE_DIR):
    long_cmd = (
        # "set -euo pipefail; "
        f"cd {remote_dir}; "
        "git pull; "
        "git submodule update --init --recursive; "
        f"source venv/activate; "
        "tt-smi -r; "
        f"{pytest_cmd}"
    )
        # "echo LOL; "
        # "sleep 3; "
        # "echo DONE; "

    # NOTE: no extra quoting around long_cmd here, we let tmux send-keys the raw string
    return f"""tmux has-session -t tests || tmux new-session -d -s tests
tmux send-keys -t tests "{long_cmd}" C-m
"""

def get_ssh_cmd(host, port=PORT):
    return [
        "ssh",
        # "-tt",
        "-p",
        str(port),
        host,
        "bash",
        "-lc",
        "cat | bash",
    ]

def get_pytest_cmd(pytest_targets, report_file):
    return f"pytest -vv --junitxml={report_file} " + " ".join([f"tests/runner/test_models.py::test_all_models[{pytest_target}]" for pytest_target in pytest_targets])

import subprocess

def tmux_session_busy(host: str, port: int = PORT, session: str = "tests") -> bool:
    """
    Returns True if the only pane in the tmux session has a running child process,
    False otherwise.

    Remote exit codes:
      19 -> busy (child process exists)
      20 -> idle (no child)
    """

    # remote script: detect busy/idle and exit with custom code
    remote_cmd = f'''
session={session}

# get pane shell pid
pane_pid=$(tmux list-panes -t "$session" -F '#{{pane_pid}}') || exit 20  # treat no session as idle

# get direct children of that shell
children=$(pgrep -P "$pane_pid")

if [ -z "$children" ]; then
    # idle
    exit 20
else
    # busy
    exit 19
fi
'''.strip()

    # build ssh command
    ssh_cmd = [
        "ssh",
        "-p", str(port),
        host,
        "bash", "-lc", shlex.quote(remote_cmd),
    ]

    # run ssh and capture only returncode
    proc = subprocess.run(ssh_cmd)
    rc = proc.returncode

    if rc == 19:
        return True   # busy
    elif rc == 20:
        return False  # idle / no session
    else:
        return True


def get_remote_script(host, pytest_targets):
    remote_dir = "/localdev/ndrakulic/tt-xla"
    venv_activate = "venv/activate"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pytest_report_file = f"/proj_sw/training_artifacts/torch_forge_models_dump/pytest_reports/{host}_{timestamp}.xml"

    pytest_cmd = get_pytest_cmd(pytest_targets, pytest_report_file)

    remote_script = make_remote_script(pytest_cmd, remote_dir)
    remote_script += " "
    return remote_script

def parse_execution_list(execution_list):
    host = {}

    num_groups = len(set([entry[-1] for entry in execution_list]))

    for entry in execution_list:
        if entry[1] not in host:
            host[entry[1]] = list([] for _ in range(num_groups))
        host[entry[1]][entry[-1]].append(entry[0])

    return host

def start_group(host, group):
    remote_script = get_remote_script(host, group)
    print(f"[{host}] starting group {group}", file=sys.stderr)

    print("===== remote script =================", file=sys.stderr)
    print(remote_script, file=sys.stderr)
    print("=================================", file=sys.stderr)

    ssh_cmd = get_ssh_cmd(host)
    print("===== ssh command =================", file=sys.stderr)
    print(ssh_cmd, file=sys.stderr)
    print("=================================", file=sys.stderr)

    proc = subprocess.Popen(
        ssh_cmd,
        stdin=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # send body to remote "cat | bash"
    proc.stdin.write(remote_script)
    proc.stdin.close()

    return proc

def main():
    execution_groups = parse_execution_list(execution_list)
    for host_name, groups in execution_groups.items():
        print(host_name)
        for group_index, group in enumerate(groups):
            print(f"Group {group_index}: {group}")

    host_states = {}
    for host, groups in execution_groups.items():
        host_states[host] = {
            "groups": groups,
            "next_idx": 0,
            "start_timestamp": datetime.now(),
        }



    # kick off first available group on each host
    for host, state in host_states.items():
        group = state["groups"][state["next_idx"]]
        start_group(host, group)
        state["next_idx"] += 1

    # main scheduler loop
    while True:
        all_done = True
        for host, state in host_states.items():
            print(f"[{host}] checking if session is busy", file=sys.stderr)
            if tmux_session_busy(host):
                print(f"[{host}] session is busy", file=sys.stderr)
                all_done = False
                continue
            print(f"[{host}] session is not busy", file=sys.stderr)
            if state["next_idx"] < len(state["groups"]):
                print(f"[{host}] starting group {state['next_idx']}", file=sys.stderr)
                group = state["groups"][state["next_idx"]]
                start_group(host, group)
                state["next_idx"] += 1
                all_done = False  # just launched new workgroup

        if all_done:
            break

        time.sleep(1)  # small backoff to avoid busy-spin

if __name__ == "__main__":
    main()
