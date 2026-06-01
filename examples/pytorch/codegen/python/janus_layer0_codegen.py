# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XLA codegen export + tt-metal post-process for Janus layer-0 LN+attn (no-dep).

Model build / CPU forward: ``janus_layer0_build`` (no ``torch_xla``).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

from janus_layer0_build import (
    DEFAULT_VARIANT,
    STAGE_NAMES,
    Layer0CodegenWorkload,
    build_layer0_codegen_workload,
    ensure_repo_on_path,
)

_TEMPLATES = Path(__file__).resolve().parent / "templates"

DEFAULT_EXPORT_PATH = "janus_layer0_ln_attn_no_dep_codegen"
_GRAPH_UTILS_OVERLAY = _TEMPLATES / "janus_graph_utils.py"
_CPU_REFERENCE_SRC = _TEMPLATES / "janus_cpu_reference"
_PATCH_GRAPH_MAIN = _TEMPLATES / "patch_graph_main.py"

__all__ = [
    "DEFAULT_EXPORT_PATH",
    "DEFAULT_VARIANT",
    "STAGE_NAMES",
    "Layer0CodegenWorkload",
    "build_layer0_codegen_workload",
    "ensure_repo_on_path",
    "export_layer0_ttnn",
    "install_tt_metal_extras",
    "run_forge_vs_cpu_stage_compare",
]


def run_forge_vs_cpu_stage_compare(
    variant: str = DEFAULT_VARIANT,
    *,
    assert_on_failure: bool = False,
) -> float:
    ensure_repo_on_path()

    from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import get_layer0_spec
    from tests.torch.models.janus_pro_pcc_drop_no_dep.op_test import (
        run_layer0_ln_attn_forge_vs_cpu_isolated,
    )
    from tests.torch.models.janus_pro_pcc_drop_no_dep.tt_device_warmup import (
        ensure_tt_device_ready,
    )

    ensure_repo_on_path()
    ensure_tt_device_ready()
    spec = get_layer0_spec(variant)
    workload = build_layer0_codegen_workload(variant)
    for line in spec.summary_lines():
        print(line)
    print(f"fixtures={workload.fixture_dir}")

    rows = run_layer0_ln_attn_forge_vs_cpu_isolated(
        f"codegen_graph_forge_vs_cpu_{variant}",
        spec,
        use_saved_inputs=True,
        load_hf_weights=False,
        assert_on_failure=assert_on_failure,
        return_metrics=True,
    )
    if rows is None:
        raise RuntimeError("run_layer0_ln_attn_forge_vs_cpu_isolated returned no metrics")
    return float(dict(rows)["self_attn"].pcc)


def _patch_graph_utils(export_path: str | Path) -> None:
    dest = Path(export_path) / "graph_0" / "utils.py"
    if not _GRAPH_UTILS_OVERLAY.is_file():
        raise FileNotFoundError(f"Missing overlay: {_GRAPH_UTILS_OVERLAY}")
    dest.write_text(_GRAPH_UTILS_OVERLAY.read_text())


def _install_cpu_reference(export_path: str | Path) -> None:
    dest = Path(export_path) / "cpu_reference"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(_CPU_REFERENCE_SRC, dest)


def _patch_graph_main(export_path: str | Path) -> None:
    spec = importlib.util.spec_from_file_location("patch_graph_main", _PATCH_GRAPH_MAIN)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_PATCH_GRAPH_MAIN}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.patch_graph_main(Path(export_path) / "graph_0" / "main.py")


def install_tt_metal_extras(export_path: str | Path) -> None:
    _patch_graph_utils(export_path)
    _install_cpu_reference(export_path)
    _patch_graph_main(export_path)


def export_layer0_ttnn(
    variant: str = DEFAULT_VARIANT,
    export_path: str | Path = DEFAULT_EXPORT_PATH,
    *,
    install_extras: bool = True,
    export_tensors: bool = True,
) -> Path:
    import torch_xla.runtime as xr
    from tt_torch import codegen_py

    workload = build_layer0_codegen_workload(variant)
    xr.set_device_type("TT")
    out = Path(export_path)
    compiler_options = {"export_tensors": True} if export_tensors else {}
    codegen_py(
        workload.wrapper,
        workload.inputs,
        export_path=str(out),
        compiler_options=compiler_options,
    )
    if install_extras:
        install_tt_metal_extras(out)
    return out
