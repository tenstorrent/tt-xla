# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CPU stacked stages — same graph as codegen / no-dep sanity (no ``torch_xla``).

Imports ``janus_layer0_build`` from tt-xla (torch + transformers + saved fixtures only).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_CODEGEN_ROOT = Path(__file__).resolve().parents[1]


def resolve_ttxla_root() -> Path:
    env = os.environ.get("JANUS_TTXLA_ROOT")
    if env:
        return Path(env)
    candidates = [
        _CODEGEN_ROOT.parent.parent.parent / "31_may_yyz" / "tt-xla",
        Path("/proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla"),
    ]
    for path in candidates:
        if (path / "examples" / "pytorch" / "codegen" / "python" / "janus_layer0_build.py").is_file():
            return path
    raise FileNotFoundError(
        "tt-xla repo not found. Set JANUS_TTXLA_ROOT (needs janus_layer0_build.py, not xla packages)."
    )


def run_layer0_ln_attn_no_dep_stacked(variant: str = "Pro_1B") -> torch.Tensor:
    repo = resolve_ttxla_root()
    codegen_dir = repo / "examples" / "pytorch" / "codegen" / "python"
    for path in (str(repo), str(codegen_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from janus_layer0_build import run_forward_stacked

    return run_forward_stacked(variant)
