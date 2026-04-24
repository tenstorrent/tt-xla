# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path


def load_compare_script():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "compare_quetzal_rewrite_ir.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_quetzal_rewrite_ir", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_export_counts_tokens_by_stage(tmp_path):
    module = load_compare_script()
    irs_dir = tmp_path / "irs"
    irs_dir.mkdir()

    (irs_dir / "shlo_model_g0_123.mlir").write_text(
        '"stablehlo.tanh" "tenstorrent.gelu_tanh"\n',
        encoding="utf-8",
    )
    (irs_dir / "ttnn_model_g0_123.mlir").write_text(
        '"ttnn.gelu" "ttnn.gelu" "ttnn.softmax"\n',
        encoding="utf-8",
    )

    summary = module.summarize_export(tmp_path)

    assert [path.name for path in summary.files] == [
        "shlo_model_g0_123.mlir",
        "ttnn_model_g0_123.mlir",
    ]
    assert summary.counts["stablehlo.tanh"] == 1
    assert summary.counts["tenstorrent.gelu_tanh"] == 1
    assert summary.counts["ttnn.gelu"] == 2
    assert summary.counts["ttnn.softmax"] == 1
    assert summary.counts_by_stage["shlo"]["tenstorrent.gelu_tanh"] == 1
    assert summary.counts_by_stage["ttnn"]["ttnn.gelu"] == 2


def test_expected_signal_present_detects_enabled_fused_tokens(tmp_path):
    module = load_compare_script()
    off_dir = tmp_path / "off" / "irs"
    on_dir = tmp_path / "on" / "irs"
    off_dir.mkdir(parents=True)
    on_dir.mkdir(parents=True)

    (off_dir / "ttnn_model_g0_123.mlir").write_text(
        '"stablehlo.tanh"\n',
        encoding="utf-8",
    )
    (on_dir / "ttnn_model_g0_123.mlir").write_text(
        '"stablehlo.tanh" "ttnn.gelu"\n',
        encoding="utf-8",
    )

    case = module.CASE_SPECS[module.CASE_DECOMPOSED_TANH_GELU]
    assert module.expected_signal_present(
        case,
        module.summarize_export(tmp_path / "off"),
        module.summarize_export(tmp_path / "on"),
    )
