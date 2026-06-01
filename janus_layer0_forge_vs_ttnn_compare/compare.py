# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Offline PCC report: Forge vs TTNN (primary), optional CPU columns.

Requires artifacts from ``capture_forge.py`` and ``capture_ttnn.py``::

  python janus_layer0_forge_vs_ttnn_compare/compare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_COMPARE_ROOT = Path(__file__).resolve().parent
if str(_COMPARE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_COMPARE_ROOT.parent))

from janus_layer0_forge_vs_ttnn_compare.metrics import print_comparison_table
from janus_layer0_forge_vs_ttnn_compare.paths import stacked_artifact


def _load(path: Path, label: str) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} artifact: {path}")
    tensor = torch.load(path, weights_only=True)
    print(f"Loaded {label}: {path}  shape={tuple(tensor.shape)}")
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare saved Forge / TTNN / CPU stacked outputs."
    )
    parser.add_argument("--variant", default="Pro_1B", choices=["Pro_1B", "Pro_7B"])
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Do not load cpu_stacked_* even if present.",
    )
    args = parser.parse_args()

    forge = _load(stacked_artifact("forge", args.variant), "forge")
    ttnn = _load(stacked_artifact("ttnn", args.variant), "ttnn")

    print("\n=== Experiment D: Forge artifact vs TTNN artifact (export fidelity) ===")
    forge_vs_ttnn = print_comparison_table(
        f"offline ({args.variant})",
        reference_label="Forge",
        actual_label="TTNN",
        reference=forge,
        actual=ttnn,
    )

    cpu_path = stacked_artifact("cpu", args.variant)
    if not args.skip_cpu and cpu_path.is_file():
        cpu = _load(cpu_path, "cpu")
        print("\n=== CPU eager vs Forge artifact (should track Experiment A ~0.77) ===")
        print_comparison_table(
            f"offline ({args.variant})",
            reference_label="CPU",
            actual_label="Forge",
            reference=cpu,
            actual=forge,
        )
        print("\n=== CPU eager vs TTNN artifact (tt-metal main.py targets this) ===")
        print_comparison_table(
            f"offline ({args.variant})",
            reference_label="CPU",
            actual_label="TTNN",
            reference=cpu,
            actual=ttnn,
        )
    elif not args.skip_cpu:
        print(f"\n(No {cpu_path.name}; run capture_cpu.py for CPU columns.)")

    attn_ft = next(m for name, m in forge_vs_ttnn if name == "self_attn")
    print(f"\nSummary: self_attn Forge vs TTNN artifact PCC = {attn_ft.pcc:.4f}")
    print("  Experiment A (live Forge vs CPU): run run_cpu_vs_forge_sanity.py — expect ~0.77")
    print("  Experiment C (tt-metal main.py): uses tensorbins; often ~0.99 vs cpu_reference")
    print("  See EXPERIMENTS.md — A and C are not the same test.")


if __name__ == "__main__":
    main()
