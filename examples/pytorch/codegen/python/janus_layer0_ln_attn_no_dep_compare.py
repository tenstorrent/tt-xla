# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gate before codegen export: isolated Forge vs CPU on the graph ``codegen_py`` will compile.

Same fair compare as ``test_layer0_ln_attn_no_dep_pro_1b``. Expect ``self_attn`` PCC ~0.99.

Usage (tt-xla root, TT device, fixtures saved)::

  python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep_compare.py
  python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep_compare.py --min-self-attn-pcc 0.95

If this passes, proceed with::

  python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from janus_layer0_codegen import (  # noqa: E402
    DEFAULT_VARIANT,
    STAGE_NAMES,
    build_layer0_codegen_workload,
    ensure_repo_on_path,
    run_forge_vs_cpu_stage_compare,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated Forge vs CPU stage PCC for the no-dep layer-0 graph used by codegen."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=["Pro_1B", "Pro_7B"],
        default=DEFAULT_VARIANT,
    )
    parser.add_argument(
        "--min-self-attn-pcc",
        type=float,
        default=0.95,
        help="Fail if self_attn PCC is below this (real Forge vs CPU regression).",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Print metrics only; do not exit non-zero on PCC floor.",
    )
    return parser.parse_args()


def main(
    variant: str = DEFAULT_VARIANT,
    *,
    min_self_attn_pcc: float = 0.95,
    gate: bool = True,
) -> float:
    ensure_repo_on_path()
    build_layer0_codegen_workload(variant)

    print(f"\n=== Codegen graph Forge vs CPU ({variant}, isolated) ===")
    print(f"Stages: {', '.join(STAGE_NAMES)}")
    print("(Fresh bundle per side — same as fixed sanity pytest.)\n")

    attn_pcc = run_forge_vs_cpu_stage_compare(variant, assert_on_failure=False)

    print(f"\n--- codegen pre-export gate ---")
    print(f"self_attn PCC = {attn_pcc:.6f}")
    print(f"minimum required: {min_self_attn_pcc:.2f} (expect ~0.99)")

    if gate and attn_pcc < min_self_attn_pcc:
        print(
            f"\nFAIL: self_attn PCC {attn_pcc:.4f} < {min_self_attn_pcc:.2f}. "
            "Investigate Forge vs CPU before export.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if gate:
        print("\nPASS: Forge matches CPU on isolated compare. Safe to run:")
        print("  python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep.py")

    return attn_pcc


if __name__ == "__main__":
    args = _parse_args()
    main(
        args.variant,
        min_self_attn_pcc=args.min_self_attn_pcc,
        gate=not args.no_gate,
    )
