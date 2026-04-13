# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone topology lookup for multi-host distributed tests.

    eval $(python scripts/multihost_topology.py --topology dual_t3k \\
        --script-dir tests/torch/multi_host/experimental)
    pytest ...
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class MultihostConfiguration:
    """
    Configuration for a specific distributed hardware topology.

    This dataclass encapsulates all the environment variables needed to configure
    the distributed runtime for a specific multi-host hardware setup. Each field
    maps to a corresponding TT_DISTRIBUTED_* environment variable.

    Attributes:
        rank_binding: The rank binding configuration name that maps to a YAML file
            containing the rank-to-device mapping for this topology. This is used
            by the MPI runtime to assign devices to processes across hosts.
            Maps to: TT_DISTRIBUTED_RANK_BINDING
            Examples: "quad_galaxy", "dual_bh_quietbox"

        controller_host_name: The hostname of the controller/coordinator node that
            orchestrates the distributed execution. This is typically the first host
            in the cluster where the main process runs.
            Maps to: TT_DISTRIBUTED_CONTROLLER_HOST_NAME
            Examples: "g05glx01", "forge-bh-01"

        hosts_list: Comma-separated list of all host machines participating in the
            distributed execution. The order should match the rank binding configuration.
            Maps to: TT_DISTRIBUTED_HOSTS_LIST
            Examples: "g05glx01,g05glx02,g05glx03,g05glx04", "forge-bh-01,forge-bh-02"
            Mutually exclusive with hosts_file; hosts_file takes precedence when set.

        hosts_file: Path to an MPI-style hostfile listing participating hosts. When set,
            TT_DISTRIBUTED_HOSTS_FILE is used instead of TT_DISTRIBUTED_HOSTS_LIST.
            Maps to: TT_DISTRIBUTED_HOSTS_FILE
            Examples: "/etc/mpirun/hostfile"

        remote_script_name: Name of the shell script used by MPI as the plm_rsh_agent
            to execute commands on remote hosts. This script handles SSH and docker
            exec commands to launch worker processes on each host.

            Only necessary for container to container tests.

            Maps to: TT_DISTRIBUTED_PLM_RSH_AGENT (via script_dir / remote_script_name)
            Default: "remote_docker.sh"

        tt_distributed_tcp_iface: The network interface name to use for MPI byte
            transport layer (BTL) TCP communication between hosts. This controls which
            network interface MPI uses for inter-host data transfer.
            Maps to: TT_DISTRIBUTED_TCP_IFACE

            "cnx1" for aus galaxies
            "enp10s0f1np1" for quietboxes
            Leave empty to use the runtime default.
    """

    rank_binding: str
    controller_host_name: str
    tt_distributed_tcp_iface: str = ""
    hosts_list: str = ""
    remote_script_name: str = "remote_docker.sh"
    hosts_file: str = ""


# Predefined topology configurations
TOPOLOGIES: Dict[str, MultihostConfiguration] = {
    "quad_galaxy": MultihostConfiguration(
        rank_binding="quad_galaxy",
        controller_host_name="g05glx01",
        hosts_list="g05glx01,g05glx02,g05glx03,g05glx04",
        tt_distributed_tcp_iface="cnx1",
    ),
    "dual_galaxy": MultihostConfiguration(
        rank_binding="dual_galaxy",
        controller_host_name="g14glx03",
        hosts_list="g14glx03,g14glx04",
        tt_distributed_tcp_iface="cnx1",
    ),
    "dual_bh_quietbox": MultihostConfiguration(
        rank_binding="dual_bh_quietbox",
        controller_host_name="forge-qbae-01",
        hosts_list="forge-qbae-01,forge-qbae-02",
        tt_distributed_tcp_iface="enp10s0f1np1",
    ),
    "dual_t3k": MultihostConfiguration(
        rank_binding="dual_t3k",
        controller_host_name="f10cs03",
        hosts_file="/etc/mpirun/hostfile",
    ),
    "dual_bh_loudbox_1x16": MultihostConfiguration(
        rank_binding="dual_bh_loudbox_1x16",
        controller_host_name="bh-lb-12",
        hosts_list="bh-lb-12,bh-lb-13",
        tt_distributed_tcp_iface="",
    ),
}


def get_distributed_worker_path() -> str:
    """
    Get path to distributed worker binary.

    First checks TT_DISTRIBUTED_WORKER_PATH environment variable.
    If not set, constructs path from TT_PJRT_PLUGIN_DIR or TTMLIR_TOOLCHAIN_DIR.

    TT_PJRT_PLUGIN_DIR is valid for wheel build. For source build, it will be set but point to invalid path.
    TT_MLIR_HOME is valid for source build inside activated venv, for local development.

    Returns:
        str: Path to the distributed worker binary

    Raises:
        AssertionError: If worker path is not set and cannot be constructed, or if path doesn't exist
    """
    worker_path = os.environ.get("TT_DISTRIBUTED_WORKER_PATH")
    if worker_path:
        assert os.path.exists(
            worker_path
        ), f"Distributed worker file does not exist at path: {worker_path} as explicitly specified in TT_DISTRIBUTED_WORKER_PATH. Please check that the path is correct."
        return worker_path

    # Auto-discover TT_PJRT_PLUGIN_DIR from pjrt_plugin_tt package if not set
    pjrt_plugin_dir = os.environ.get("TT_PJRT_PLUGIN_DIR")
    if not pjrt_plugin_dir:
        try:
            from pjrt_plugin_tt import setup_tt_pjrt_plugin_dir
            setup_tt_pjrt_plugin_dir()
            pjrt_plugin_dir = os.environ.get("TT_PJRT_PLUGIN_DIR")
        except ImportError:
            pass  # pjrt_plugin_tt not installed, will fall back to TT_MLIR_HOME

    if pjrt_plugin_dir:
        worker_path = os.path.join(
            pjrt_plugin_dir, "bin/ttmlir/runtime/distributed/worker"
        )
        if os.path.exists(worker_path):
            return worker_path

    tt_mlir_home_dir = os.environ.get("TT_MLIR_HOME")
    assert tt_mlir_home_dir, (
        "Neither TT_DISTRIBUTED_WORKER_PATH, TT_PJRT_PLUGIN_DIR, nor TTMLIR_TOOLCHAIN_DIR "
        "environment variable is set. The path to the distributed worker binary cannot be constructed."
    )

    worker_path = str(
        Path(tt_mlir_home_dir) / "build" / "runtime" / "bin" / "distributed" / "worker"
    )
    assert os.path.exists(
        worker_path
    ), f"Distributed worker file does not exist at path: {worker_path}, based on TT_MLIR_HOME."

    return worker_path


def get_env_vars(topology: str, script_dir: Path) -> Dict[str, str]:
    """
    Return the dict of environment variables required for the given topology.

    This is the core logic shared by both the pytest fixture (conftest.py) and
    the CLI entry point below. Callers are responsible for applying the variables
    to os.environ if needed.

    Args:
        topology: Key into TOPOLOGIES, e.g. "dual_t3k"
        script_dir: Directory that contains the remote_docker.sh helper script

    Returns:
        Dict mapping env var names to their string values

    Raises:
        ValueError: If topology is not in TOPOLOGIES
    """
    if topology not in TOPOLOGIES:
        raise ValueError(
            f"Unknown topology '{topology}'. Available: {list(TOPOLOGIES.keys())}"
        )

    topo = TOPOLOGIES[topology]
    env_vars: Dict[str, str] = {
        "TT_RUNTIME_ENABLE_DISTRIBUTED": "1",
        "TT_DISTRIBUTED_WORKER_PATH": get_distributed_worker_path(),
        "TT_DISTRIBUTED_RANK_BINDING": topo.rank_binding,
        "TT_DISTRIBUTED_CONTROLLER_HOST_NAME": topo.controller_host_name,
    }

    if topo.tt_distributed_tcp_iface:
        env_vars["TT_DISTRIBUTED_TCP_IFACE"] = topo.tt_distributed_tcp_iface

    # hosts_file takes precedence over hosts_list when both are set
    if topo.hosts_file:
        env_vars["TT_DISTRIBUTED_HOSTS_FILE"] = topo.hosts_file
    elif topo.hosts_list:
        env_vars["TT_DISTRIBUTED_HOSTS_LIST"] = topo.hosts_list

    # TT_DISTRIBUTED_PLM_RSH_AGENT is only needed for container to container tests.
    # MPI's plm_ssh_agent requires an absolute path, so resolve() here regardless
    # of whether script_dir was passed as relative or absolute.
    if topo.remote_script_name:
        env_vars["TT_DISTRIBUTED_PLM_RSH_AGENT"] = str(
            script_dir.resolve() / topo.remote_script_name
        )

    return env_vars


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Print shell export statements for a given distributed topology. "
            "Intended to be eval'd in CI bash steps: "
            "eval $(python -m tests.torch.multi_host.topology --topology dual_t3k)"
        )
    )
    parser.add_argument(
        "--topology",
        required=True,
        choices=list(TOPOLOGIES.keys()),
        help="Topology name to configure",
    )
    parser.add_argument(
        "--script-dir",
        default=str(Path(__file__).parent / "experimental"),
        help=(
            "Directory containing remote_docker.sh "
            "(default: experimental/ next to this file)"
        ),
    )
    args = parser.parse_args()

    try:
        env_vars = get_env_vars(args.topology, Path(args.script_dir))
    except (ValueError, AssertionError) as exc:
        print(f"echo 'topology.py error: {exc}' >&2", file=sys.stderr)
        sys.exit(1)

    for key, value in env_vars.items():
        # Single-quote the value so special chars are safe in the shell
        escaped = value.replace("'", "'\\''")
        print(f"export {key}='{escaped}'")
