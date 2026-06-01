# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Artifact paths shared by capture / compare scripts."""

from __future__ import annotations

import os
from pathlib import Path

_COMPARE_ROOT = Path(__file__).resolve().parent


def artifacts_dir() -> Path:
    env = os.environ.get("JANUS_COMPARE_ARTIFACTS_DIR")
    if env:
        return Path(env)
    return _COMPARE_ROOT / "artifacts"


def stacked_artifact(backend: str, variant: str = "Pro_1B") -> Path:
    slug = variant.lower().replace("_", "")
    return artifacts_dir() / f"{backend}_stacked_{slug}.pt"


def default_ttnn_graph_dir() -> Path:
    env = os.environ.get("JANUS_TTNN_GRAPH_DIR")
    if env:
        return Path(env)
    candidates = [
        _COMPARE_ROOT.parent.parent
        / "31_may_tt_metal"
        / "tt-metal"
        / "janus_layer0_ln_attn_no_dep_codegen"
        / "graph_0",
        Path(
            "/proj_sw/user_dev/ctr-akannan/31_may_tt_metal/tt-metal/"
            "janus_layer0_ln_attn_no_dep_codegen/graph_0"
        ),
        _COMPARE_ROOT.parent
        / "janus_layer0_ln_attn_no_dep_codegen"
        / "graph_0",
    ]
    for path in candidates:
        if (path / "main.py").is_file():
            return path
    raise FileNotFoundError(
        "TTNN graph_0 not found. Set JANUS_TTNN_GRAPH_DIR to "
        "janus_layer0_ln_attn_no_dep_codegen/graph_0"
    )
