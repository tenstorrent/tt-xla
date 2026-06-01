# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Save CPU eager golden (optional third column in compare report).

Usage (any env with torch + transformers + fixtures)::

  python janus_layer0_forge_vs_ttnn_compare/capture_cpu.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_COMPARE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _COMPARE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from janus_layer0_forge_vs_ttnn_compare.paths import artifacts_dir, stacked_artifact


def _run_cpu_stacked(variant: str) -> torch.Tensor:
    """Same CPU as codegen / sanity (no torch_xla)."""
    codegen_py = _REPO_ROOT / "examples" / "pytorch" / "codegen" / "python"
    if str(codegen_py) not in sys.path:
        sys.path.insert(0, str(codegen_py))
    from janus_layer0_build import run_forward_stacked

    return run_forward_stacked(variant)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture CPU eager stacked stages.")
    parser.add_argument("--variant", default="Pro_1B", choices=["Pro_1B", "Pro_7B"])
    args = parser.parse_args()

    out_path = stacked_artifact("cpu", args.variant)
    artifacts_dir().mkdir(parents=True, exist_ok=True)

    print(f"Running CPU reference ({args.variant}) ...")
    stacked = _run_cpu_stacked(args.variant)
    torch.save(stacked, out_path)
    print(f"Wrote {out_path}  shape={tuple(stacked.shape)} dtype={stacked.dtype}")


if __name__ == "__main__":
    main()
