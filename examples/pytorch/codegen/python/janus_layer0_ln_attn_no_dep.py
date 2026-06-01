# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Export TTNN / EmitPy artifacts for Janus layer-0 LN+attn (no-dep, saved fixtures).

Run **after** isolated Forge-vs-CPU compare passes (~0.99 on ``self_attn``)::

  python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep_compare.py
  python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep.py

Prerequisites (once per variant)::

  pytest -s tests/torch/models/janus_pro_pcc_drop_no_dep/test_save_layer0_no_dep_fixtures.py::test_save_layer0_no_dep_fixtures_pro_1b
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from janus_layer0_codegen import (
    DEFAULT_EXPORT_PATH,
    DEFAULT_VARIANT,
    export_layer0_ttnn,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Codegen export for no-dep Janus layer-0 input_layernorm + self_attn.",
    )
    parser.add_argument(
        "--variant",
        choices=["Pro_1B", "Pro_7B"],
        default=DEFAULT_VARIANT,
    )
    parser.add_argument(
        "--export-path",
        default=DEFAULT_EXPORT_PATH,
        help="Output directory for codegen artifacts.",
    )
    parser.add_argument(
        "--no-tt-metal-extras",
        action="store_true",
        help="Skip utils overlay, cpu_reference copy, and main.py patch.",
    )
    parser.add_argument(
        "--no-export-tensors",
        action="store_true",
        help="Do not pass export_tensors=1 to the compiler.",
    )
    return parser.parse_args()


def main(
    variant: str = DEFAULT_VARIANT,
    export_path: str = DEFAULT_EXPORT_PATH,
    *,
    install_extras: bool = True,
    export_tensors: bool = True,
) -> Path:
    out = export_layer0_ttnn(
        variant,
        export_path,
        install_extras=install_extras,
        export_tensors=export_tensors,
    )
    print(f"Codegen export written to {out.resolve()}")
    return out


def test_janus_layer0_ln_attn_no_dep_codegen() -> None:
    """Codegen creates output folder when Pro_1B fixtures exist."""
    import pytest

    from janus_layer0_codegen import build_layer0_codegen_workload

    try:
        build_layer0_codegen_workload("Pro_1B")
    except FileNotFoundError:
        pytest.skip("Pro_1B fixtures missing; run test_save_layer0_no_dep_fixtures_pro_1b first")

    output_dir = Path(DEFAULT_EXPORT_PATH)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        main()
        assert output_dir.is_dir()
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    args = _parse_args()
    main(
        args.variant,
        args.export_path,
        install_extras=not args.no_tt_metal_extras,
        export_tensors=not args.no_export_tensors,
    )
