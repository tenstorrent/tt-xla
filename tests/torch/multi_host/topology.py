# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Re-export multihost topology helpers from ``scripts/multihost_topology.py``.

Keeping the implementation under ``scripts/`` avoids registering a top-level
``torch`` package from ``tests/torch/__init__.py``, which would shadow the real
PyTorch when ``$(pwd)/tests`` is on ``PYTHONPATH`` (see ``venv/activate``).

CLI users should run::

    python scripts/multihost_topology.py --topology <name>

from the repository root instead of ``python -m tests.torch.multi_host.topology``.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "multihost_topology.py"

_spec = importlib.util.spec_from_file_location("multihost_topology", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MultihostConfiguration = _mod.MultihostConfiguration
TOPOLOGIES = _mod.TOPOLOGIES
get_distributed_worker_path = _mod.get_distributed_worker_path
get_env_vars = _mod.get_env_vars

__all__ = [
    "MultihostConfiguration",
    "TOPOLOGIES",
    "get_distributed_worker_path",
    "get_env_vars",
]

if __name__ == "__main__":
    # Delegate to the canonical script so `python tests/torch/multi_host/topology.py` still works.
    sys.argv[0] = str(_SCRIPT)
    runpy.run_path(str(_SCRIPT), run_name="__main__")
