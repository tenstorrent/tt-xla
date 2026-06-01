# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Run exported TTNN ``graph_0/main.py`` and save stacked stages.

Usage (tt-metal ``python_env``, device available)::

  cd /proj_sw/user_dev/ctr-akannan/31_may_tt_metal/tt-metal
  python /proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla/janus_layer0_forge_vs_ttnn_compare/capture_ttnn.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_COMPARE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _COMPARE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_graph_main(graph_dir: Path):
    codegen_root = graph_dir.parent
    for path in (graph_dir, codegen_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import main as graph_main  # noqa: PLC0415

    return graph_main


def capture_ttnn_stacked(graph_dir: Path) -> torch.Tensor:
    import ttnn

    graph_main = _import_graph_main(graph_dir)
    activations = graph_main.load_activations_for__main()
    weights = graph_main.load_weights_for__main()
    outputs = graph_main._main(activations, weights)
    stacked = outputs[0]
    if hasattr(stacked, "cpu"):
        stacked = ttnn.to_torch(stacked)
    stacked = stacked.detach().cpu().to(torch.bfloat16)
    if stacked.ndim != 4 or stacked.shape[0] != 3:
        raise ValueError(f"Expected [3, B, S, H] from TTNN, got {tuple(stacked.shape)}")
    return stacked


def main() -> None:
    from janus_layer0_forge_vs_ttnn_compare.paths import (
        artifacts_dir,
        default_ttnn_graph_dir,
        stacked_artifact,
    )

    parser = argparse.ArgumentParser(description="Capture TTNN codegen stacked stages.")
    parser.add_argument("--variant", default="Pro_1B", choices=["Pro_1B", "Pro_7B"])
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Path to graph_0/ (default: JANUS_TTNN_GRAPH_DIR or auto-detect)",
    )
    args = parser.parse_args()

    graph_dir = args.graph_dir or default_ttnn_graph_dir()
    out_path = stacked_artifact("ttnn", args.variant)
    artifacts_dir().mkdir(parents=True, exist_ok=True)

    print(f"Running TTNN graph at {graph_dir} ...")
    stacked = capture_ttnn_stacked(graph_dir)
    torch.save(stacked, out_path)
    print(f"Wrote {out_path}  shape={tuple(stacked.shape)} dtype={stacked.dtype}")


if __name__ == "__main__":
    main()
