# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for onnx-mlir StableHLO canonicalization."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python_package") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python_package"))

from tt_onnx.mlir_utils import (  # noqa: E402
    _rewrite_dots_to_dot_general,
    _rewrite_real_dynamic_slice_to_slice,
    _rewrite_torch_index_select_to_gather,
)


def test_rewrite_rank2_dot_to_dot_general():
    before = (
        "    %2 = stablehlo.dot %0, %1 : "
        "(tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>"
    )
    after = _rewrite_dots_to_dot_general(before)
    assert "stablehlo.dot_general" in after
    assert "batching_dims" not in after
    assert "contracting_dims = [1] x [0]" in after
    assert "stablehlo.dot" not in after.replace("stablehlo.dot_general", "")


def test_rewrite_rank3_dot_to_dot_general():
    before = (
        "    %0 = stablehlo.dot %arg0, %arg1 : "
        "(tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>"
    )
    after = _rewrite_dots_to_dot_general(before)
    assert "batching_dims = [0] x [0]" in after
    assert "contracting_dims = [2] x [1]" in after


def test_rewrite_static_real_dynamic_slice_to_slice():
    before = """
    %starts = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
    %limits = stablehlo.constant dense<[1, 3]> : tensor<2xi64>
    %strides = stablehlo.constant dense<[1, 1]> : tensor<2xi64>
    %out = stablehlo.real_dynamic_slice %arg0, %starts, %limits, %strides : (tensor<1x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
"""
    after = _rewrite_real_dynamic_slice_to_slice(before)
    assert "stablehlo.slice %arg0 [0:1:1, 1:3:1]" in after
    assert "real_dynamic_slice" not in after


def test_strip_onnx_no_value_lines():
    from tt_onnx.mlir_utils import _strip_onnx_entry_point  # noqa: PLC0415

    before = """
    %bias = "onnx.NoValue"() : () -> none
    %0 = stablehlo.add %arg0, %arg1 : tensor<1x4xf32>
    """
    after = _strip_onnx_entry_point(before)
    assert "NoValue" not in after
    assert "stablehlo.add" in after


def test_rewrite_torch_index_select_to_gather_axis0():
    before = (
        '    %5 = "stablehlo.torch_index_select"(%arg0, %4) '
        "{batch_dims = 0 : i64, dim = 0 : i64} "
        ": (tensor<3x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>"
    )
    after = _rewrite_torch_index_select_to_gather(before)
    assert "torch_index_select" not in after
    assert "stablehlo.gather %arg0" in after
    assert "stablehlo.reshape %4" in after
    assert "offset_dims = [1]" in after
    assert "collapsed_slice_dims = [0]" in after
    assert "start_index_map = [0]" in after
    assert "slice_sizes = dense<[1, 4]>" in after


def test_rewrite_onnx_mlir_gather_full_chain():
    # From tools/onnx/build/tt_onnx/triage/gather/gather.stablehlo.raw.mlir
    before = """
    %4 = stablehlo.select %2, %3, %arg1 : tensor<2xi1>, tensor<2xi64>
    %5 = "stablehlo.torch_index_select"(%arg0, %4) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<3x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>
"""
    after = _rewrite_torch_index_select_to_gather(before)
    assert "torch_index_select" not in after
    assert "stablehlo.gather %arg0, %gather_indices" in after


def test_rewrite_onnx_mlir_slice_select_chain():
    # Truncated from onnx-mlir Slice.mlir CHECK (positive step, static params).
    before = """
    %c0 = stablehlo.constant dense<0> : tensor<1xi64>
    %c1 = stablehlo.constant dense<1> : tensor<1xi64>
    %c4 = stablehlo.constant dense<4> : tensor<1xi64>
    %starts = stablehlo.constant dense<[1]> : tensor<1xi64>
    %ends = stablehlo.constant dense<[3]> : tensor<1xi64>
    %steps = stablehlo.constant dense<[1]> : tensor<1xi64>
    %neg_step = stablehlo.compare  LT, %steps, %c0,  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %begin = stablehlo.select %neg_step, %c0, %starts : tensor<1xi1>, tensor<1xi64>
    %end_cmp = stablehlo.compare  GT, %ends, %c4,  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %end = stablehlo.select %end_cmp, %c4, %ends : tensor<1xi1>, tensor<1xi64>
    %neg_begin = stablehlo.compare  LT, %begin, %c0,  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %begin_plus = stablehlo.add %begin, %c4 : tensor<1xi64>
    %begin_fix = stablehlo.select %neg_begin, %begin_plus, %begin : tensor<1xi1>, tensor<1xi64>
    %start0 = stablehlo.constant dense<0> : tensor<1xi64>
    %limit0 = stablehlo.constant dense<1> : tensor<1xi64>
    %stride0 = stablehlo.constant dense<1> : tensor<1xi64>
    %start_indices = stablehlo.concatenate %start0, %begin_fix, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %limit_indices = stablehlo.concatenate %limit0, %end, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %stride_indices = stablehlo.concatenate %stride0, %steps, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %out = stablehlo.real_dynamic_slice %arg0, %start_indices, %limit_indices, %stride_indices : (tensor<1x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
"""
    after = _rewrite_real_dynamic_slice_to_slice(before)
    assert "stablehlo.slice %arg0 [0:1:1, 1:3:1]" in after
