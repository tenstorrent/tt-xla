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

_TT_METAL_CPU_REF = (
    _REPO_ROOT.parent.parent
    / "31_may_tt_metal"
    / "tt-metal"
    / "janus_layer0_ln_attn_no_dep_codegen"
)
if _TT_METAL_CPU_REF.is_dir() and str(_TT_METAL_CPU_REF) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_CPU_REF))


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture CPU eager stacked stages.")
    parser.add_argument("--variant", default="Pro_1B", choices=["Pro_1B", "Pro_7B"])
    args = parser.parse_args()

    from cpu_reference.forward import run_forward_from_fixtures

    out_path = stacked_artifact("cpu", args.variant)
    artifacts_dir().mkdir(parents=True, exist_ok=True)

    print(f"Running CPU reference ({args.variant}) ...")
    stacked = run_forward_from_fixtures(args.variant)
    torch.save(stacked, out_path)
    print(f"Wrote {out_path}  shape={tuple(stacked.shape)} dtype={stacked.dtype}")


if __name__ == "__main__":
    main()
