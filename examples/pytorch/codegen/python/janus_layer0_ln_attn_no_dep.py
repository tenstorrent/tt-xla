# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Export TT codegen / TTNN op sequence for the canonical layer-0 LN+attn PCC-drop graph.

Uses ``Layer0LnAttnNoDep`` (HF ``LlamaRMSNorm`` + ``LlamaAttention`` + saved fixtures).
No ``tt_forge_models`` / Janus checkpoint load — only ``torch``, ``transformers``, and files under
``janus_logs/layer0_tensors/<variant>/``.

Prerequisites (once per variant)::

  pytest -s tests/torch/models/janus_pro_pcc_drop_no_dep/test_save_layer0_no_dep_fixtures.py::test_save_layer0_no_dep_fixtures_pro_1b

Usage::

  python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep.py
  python examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep.py --variant Pro_7B
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py

# Repo root on sys.path when run as a script from tt-xla.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import get_layer0_spec
from tests.torch.models.janus_pro_pcc_drop_no_dep.build_modules import (
    Layer0LnAttnNoDep,
    build_layer0_no_dep,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.saved_fixtures import (
    fixture_dir_for_variant,
    saved_fixtures_available,
)

DEFAULT_EXPORT_PATH = "janus_layer0_ln_attn_no_dep_codegen"
_GRAPH_UTILS_OVERLAY = (
    Path(__file__).resolve().parent / "templates" / "janus_graph_utils.py"
)
_CPU_REFERENCE_SRC = (
    Path(__file__).resolve().parent / "templates" / "janus_cpu_reference"
)


def _patch_graph_utils(export_path: str) -> None:
    """EmitPy overwrites utils.py; restore resolve_tensor_path for tt-metal root runs."""
    dest = Path(export_path) / "graph_0" / "utils.py"
    if not _GRAPH_UTILS_OVERLAY.is_file():
        raise FileNotFoundError(f"Missing overlay: {_GRAPH_UTILS_OVERLAY}")
    dest.write_text(_GRAPH_UTILS_OVERLAY.read_text())


def _install_cpu_reference(export_path: str) -> None:
    """Same CPU golden path as ``Layer0LnAttnNoDep`` used in ``codegen_py``."""
    dest = Path(export_path) / "cpu_reference"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(_CPU_REFERENCE_SRC, dest)


def _patch_graph_main(export_path: str) -> None:
    import importlib.util

    patch_path = Path(__file__).resolve().parent / "templates" / "patch_graph_main.py"
    spec = importlib.util.spec_from_file_location("patch_graph_main", patch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {patch_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.patch_graph_main(Path(export_path) / "graph_0" / "main.py")


def _install_tt_metal_extras(export_path: str) -> None:
    _patch_graph_utils(export_path)
    _install_cpu_reference(export_path)
    _patch_graph_main(export_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Codegen for no-dep Janus layer-0 fused input_layernorm + self_attn "
            "(stacked decode_setup / LN / attn outputs, ~0.77 self_attn PCC on TT)."
        )
    )
    parser.add_argument(
        "--variant",
        choices=["Pro_1B", "Pro_7B"],
        default="Pro_1B",
    )
    parser.add_argument(
        "--export-path",
        default=DEFAULT_EXPORT_PATH,
        help="Output directory for codegen artifacts (TTNN op dumps when export_tensors=1).",
    )
    return parser.parse_args()


def main(variant_name: str = "Pro_1B", export_path: str = DEFAULT_EXPORT_PATH) -> None:
    if not saved_fixtures_available(variant_name):
        raise FileNotFoundError(
            f"Missing fixtures under {fixture_dir_for_variant(variant_name)}. Run:\n"
            "  pytest -s tests/torch/models/janus_pro_pcc_drop_no_dep/"
            f"test_save_layer0_no_dep_fixtures.py::test_save_layer0_no_dep_fixtures_{variant_name.lower()}"
        )

    spec = get_layer0_spec(variant_name)
    bundle = build_layer0_no_dep(
        spec,
        use_saved_inputs=True,
        load_hf_weights=False,
    )
    wrapper = Layer0LnAttnNoDep(bundle)
    inputs = bundle.inputs_embeds

    xr.set_device_type("TT")
    codegen_py(
        wrapper,
        inputs,
        export_path=export_path,
        compiler_options={
            "export_tensors": True,
        },
    )
    _install_tt_metal_extras(export_path)


def test_janus_layer0_ln_attn_no_dep_codegen() -> None:
    """Codegen creates output folder when Pro_1B fixtures exist."""
    import pytest

    if not saved_fixtures_available("Pro_1B"):
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
    main(args.variant, args.export_path)
