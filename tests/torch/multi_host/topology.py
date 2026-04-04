# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Re-exports from scripts/multihost_topology.py (the canonical location)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "multihost_topology",
    Path(__file__).parents[3] / "scripts" / "multihost_topology.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MultihostConfiguration = _mod.MultihostConfiguration
TOPOLOGIES = _mod.TOPOLOGIES
get_distributed_worker_path = _mod.get_distributed_worker_path
get_env_vars = _mod.get_env_vars
