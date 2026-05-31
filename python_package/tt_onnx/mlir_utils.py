# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Normalize onnx-mlir StableHLO for JAX/tt-xla compilation."""

from __future__ import annotations

import re

_FUNC_LINE_RE = re.compile(
    r"^\s*func\.func\s+(@[\w.]+)\((.*)\)\s*->\s*(\(.*\)|tensor<[^>]+>)\s*\{\s*$",
    re.MULTILINE,
)
_TENSOR_TYPE_RE = re.compile(r"tensor<[^>]+>")
# JAX/XLA compile path requires the public entry point to be named @main.
_ENTRY_FUNC_NAME = "main"


def canonicalize_onnx_stablehlo(mlir_text: str) -> str:
    """
    Prepare onnx-mlir StableHLO for JAX's MLIR parser + tt-xla PJRT.

    onnx-mlir emits func/arith/shape/onnx ops that jaxlib's Python MLIR context
    does not register. For the common static-shape elementwise case we rewrite to
    a minimal stablehlo+func module. EntryPoint metadata is always stripped.
    """
    text = _strip_onnx_entry_point(mlir_text)
    simplified = _simplify_static_elementwise_add(text)
    if simplified is not None:
        return _apply_stablehlo_rewrites(simplified)
    text = _ensure_module_sym_name_attr(text)
    text = _rename_entry_function_to_main(text)
    text = _mark_entry_function_public(text)
    return _apply_stablehlo_rewrites(text)


def _strip_onnx_entry_point(mlir_text: str) -> str:
    lines = [
        line
        for line in mlir_text.splitlines()
        if "onnx.EntryPoint" not in line
    ]
    return "\n".join(lines).strip() + "\n"


def _simplify_static_elementwise_add(mlir_text: str) -> str | None:
    """Collapse shape-broadcast prelude into stablehlo.add for matched static shapes."""
    if "stablehlo.add" not in mlir_text:
        return None
    if "shape." not in mlir_text and "arith." not in mlir_text:
        return None

    match = _FUNC_LINE_RE.search(mlir_text)
    if match is None:
        return None

    func_name, args_blob, result_blob = match.groups()
    del func_name  # onnx-mlir uses @main_graph; XLA expects @main.
    arg_types = _TENSOR_TYPE_RE.findall(args_blob)
    if len(arg_types) < 2:
        return None
    if arg_types[0] != arg_types[1]:
        return None

    out_types = _TENSOR_TYPE_RE.findall(result_blob)
    if not out_types:
        return None
    out_type = out_types[0]
    elem_type = arg_types[0]

    return f"""module attributes {{sym_name = "onnx_module"}} {{
  func.func public @{_ENTRY_FUNC_NAME}(%arg0: {elem_type}, %arg1: {elem_type}) -> {out_type} {{
    %0 = stablehlo.add %arg0, %arg1 : {elem_type}
    return %0 : {out_type}
  }}
}}
"""


def _rename_entry_function_to_main(mlir_text: str) -> str:
    """Rename the first func.func symbol to @main for JAX/XLA ingestion."""
    return re.sub(
        r"^(\s*func\.func\s+(?:public\s+)?)@\w+",
        rf"\1@{_ENTRY_FUNC_NAME}",
        mlir_text,
        count=1,
        flags=re.MULTILINE,
    )


def _mark_entry_function_public(mlir_text: str) -> str:
    """tt-xla ModuleBuilder requires exactly one public func.func entry point."""
    if re.search(rf"func\.func\s+public\s+@{_ENTRY_FUNC_NAME}\b", mlir_text):
        return mlir_text
    updated, count = re.subn(
        rf"^(\s*func\.func)\s+(@{_ENTRY_FUNC_NAME}\b)",
        r"\1 public \2",
        mlir_text,
        count=1,
        flags=re.MULTILINE,
    )
    if count:
        return updated
    updated, count = re.subn(
        r"^(\s*func\.func)\s+(@\w+)",
        rf"\1 public @{_ENTRY_FUNC_NAME}",
        mlir_text,
        count=1,
        flags=re.MULTILINE,
    )
    return updated if count else mlir_text


def _ensure_module_sym_name_attr(mlir_text: str) -> str:
    if 'sym_name = "onnx_module"' in mlir_text or "sym_name = 'onnx_module'" in mlir_text:
        return mlir_text
    if re.match(r"^\s*module\s+attributes\s+\{", mlir_text):
        return re.sub(
            r"^(\s*module\s+attributes\s+\{)",
            r'\1 sym_name = "onnx_module",',
            mlir_text,
            count=1,
            flags=re.MULTILINE,
        )
    return f'module attributes {{sym_name = "onnx_module"}} {{\n{mlir_text}\n}}\n'


# stablehlo.dot (onnx-mlir 2D MatMul/Gemm) -> stablehlo.dot_general for tt-mlir.
_DOT_LINE_RE = re.compile(
    r"(?P<prefix>^\s*(?:%[\w.]+\s*=\s*)?)"
    r"stablehlo\.dot\s+(?P<lhs>%[\w.]+),\s*(?P<rhs>%[\w.]+)\s*"
    r":\s*\((?P<lhs_ty>tensor<[^>]+>),\s*(?P<rhs_ty>tensor<[^>]+>)\)\s*"
    r"->\s*(?P<out_ty>tensor<[^>]+>)"
    r"(?P<suffix>\s*(?:,\s*precision\s*=\s*\[[^\]]*\])?\s*(?:loc\([^)]*\))?)?\s*$",
    re.MULTILINE,
)


def _tensor_rank(tensor_type: str) -> int:
    match = re.search(r"tensor<([^>]+)>", tensor_type)
    if match is None:
        return 0
    parts = match.group(1).split("x")
    return max(len(parts) - 1, 0)


def _dot_to_dot_general_dims(rank: int) -> str:
    """Build dot_general dimension attrs matching StableHLO dot semantics."""
    if rank < 2:
        raise ValueError(f"stablehlo.dot expects rank >= 2, got rank {rank}")
    batch = ", ".join(str(i) for i in range(rank - 2))
    parts: list[str] = []
    if rank > 2:
        parts.append(f"batching_dims = [{batch}] x [{batch}]")
    parts.append(f"contracting_dims = [{rank - 1}] x [{rank - 2}]")
    return ", ".join(parts)


def _rewrite_dots_to_dot_general(mlir_text: str) -> str:
    """
    Rewrite stablehlo.dot to stablehlo.dot_general.

    onnx-mlir emits dot for rank-2 MatMul/Gemm; tt-mlir lowers dot_general only.
    Semantics match StableHLO: batch dims [0..R-2], contract lhs [R-1], rhs [R-2].
    """

    def _replace(match: re.Match[str]) -> str:
        lhs_ty = match.group("lhs_ty")
        rhs_ty = match.group("rhs_ty")
        rank = _tensor_rank(lhs_ty)
        if rank != _tensor_rank(rhs_ty) or rank < 2:
            return match.group(0)
        dims = _dot_to_dot_general_dims(rank)
        return (
            f"{match.group('prefix')}stablehlo.dot_general "
            f"{match.group('lhs')}, {match.group('rhs')}, {dims} "
            f": ({lhs_ty}, {rhs_ty}) -> {match.group('out_ty')}"
            f"{match.group('suffix') or ''}"
        )

    return _DOT_LINE_RE.sub(_replace, mlir_text)


def _apply_stablehlo_rewrites(mlir_text: str) -> str:
    mlir_text = _rewrite_dots_to_dot_general(mlir_text)
    return _rewrite_real_dynamic_slice_to_slice(mlir_text)


_SSA_DEF_RE = re.compile(r"^\s*(%[\w.]+)\s*=\s*(.+)$")
_RDS_LINE_RE = re.compile(
    r"^(?P<prefix>\s*(?P<result>%[\w.]+)\s*=\s*)"
    r"stablehlo\.real_dynamic_slice\s+"
    r"(?P<operand>%[\w.]+),\s*"
    r"(?P<start>%[\w.]+),\s*"
    r"(?P<limit>%[\w.]+),\s*"
    r"(?P<strides>%[\w.]+)\s*"
    r":\s*\((?P<in_tys>[^)]+)\)\s*->\s*(?P<out_ty>tensor<[^>]+>)"
    r"(?P<suffix>\s*)$",
    re.MULTILINE,
)


def _parse_dense_int_list(dense: str) -> list[int]:
    dense = dense.strip()
    if dense.startswith("["):
        values: list[int] = []

        def _flatten(item: object) -> None:
            if isinstance(item, int):
                values.append(item)
            elif isinstance(item, list):
                for child in item:
                    _flatten(child)
            else:
                raise ValueError(f"unsupported dense literal: {dense!r}")

        import ast

        parsed = ast.literal_eval(dense)
        _flatten(parsed)
        return values
    token = dense.split()[0]
    return [int(token.replace("i64", ""))]


def _build_ssa_def_map(mlir_text: str) -> dict[str, str]:
    defs: dict[str, str] = {}
    for line in mlir_text.splitlines():
        match = _SSA_DEF_RE.match(line)
        if match:
            defs[match.group(1)] = match.group(2).strip()
    return defs


def _eval_i64_tensor(
    defs: dict[str, str],
    ssa: str,
    memo: dict[str, list[int]],
    visiting: set[str],
) -> list[int] | None:
    if ssa in memo:
        return memo[ssa]
    if ssa in visiting:
        return None
    visiting.add(ssa)
    rhs = defs.get(ssa)
    if rhs is None:
        visiting.discard(ssa)
        return None

    try:
        match = re.match(
            r"stablehlo\.constant\s+dense<([^>]+)>\s*:\s*tensor(?:<[^>]*i64[^>]*>)?",
            rhs,
        )
        if match:
            memo[ssa] = _parse_dense_int_list(match.group(1))
            return memo[ssa]

        match = re.match(
            r"stablehlo\.slice\s+(%[\w.]+)\s*\[(\d+):(\d+)\]\s*:",
            rhs,
        )
        if match:
            src = _eval_i64_tensor(defs, match.group(1), memo, visiting)
            if src is None:
                return None
            begin, end = int(match.group(2)), int(match.group(3))
            memo[ssa] = src[begin:end]
            return memo[ssa]

        match = re.match(
            r"stablehlo\.concatenate\s+((?:%[\w.]+(?:,\s*)?)+)\s*,\s*dim\s*=\s*\d+",
            rhs,
        )
        if match:
            parts = re.findall(r"%[\w.]+", match.group(1))
            merged: list[int] = []
            for part in parts:
                values = _eval_i64_tensor(defs, part, memo, visiting)
                if values is None:
                    return None
                merged.extend(values)
            memo[ssa] = merged
            return memo[ssa]

        match = re.match(
            r"stablehlo\.select\s+(%[\w.]+),\s*(%[\w.]+),\s*(%[\w.]+)\s*:",
            rhs,
        )
        if match:
            pred, on_true, on_false = match.groups()
            pred_bits = _eval_i1_tensor(defs, pred, memo, visiting)
            true_vals = _eval_i64_tensor(defs, on_true, memo, visiting)
            false_vals = _eval_i64_tensor(defs, on_false, memo, visiting)
            if pred_bits is None or true_vals is None or false_vals is None:
                return None
            if len(pred_bits) == 1 and len(true_vals) == 1 and len(false_vals) == 1:
                memo[ssa] = true_vals if pred_bits[0] else false_vals
                return memo[ssa]
            if len(pred_bits) != len(true_vals) or len(pred_bits) != len(false_vals):
                return None
            memo[ssa] = [
                true_vals[i] if pred_bits[i] else false_vals[i]
                for i in range(len(pred_bits))
            ]
            return memo[ssa]

        match = re.match(r"stablehlo\.add\s+(%[\w.]+),\s*(%[\w.]+)\s*:", rhs)
        if match:
            lhs = _eval_i64_tensor(defs, match.group(1), memo, visiting)
            rhs_vals = _eval_i64_tensor(defs, match.group(2), memo, visiting)
            if lhs is None or rhs_vals is None or len(lhs) != len(rhs_vals):
                return None
            memo[ssa] = [lhs[i] + rhs_vals[i] for i in range(len(lhs))]
            return memo[ssa]

        match = re.match(r"stablehlo\.negate\s+(%[\w.]+)\s*:", rhs)
        if match:
            src = _eval_i64_tensor(defs, match.group(1), memo, visiting)
            if src is None:
                return None
            memo[ssa] = [-v for v in src]
            return memo[ssa]
    finally:
        visiting.discard(ssa)
    return None


def _eval_i1_tensor(
    defs: dict[str, str],
    ssa: str,
    memo_i64: dict[str, list[int]],
    visiting: set[str],
) -> list[bool] | None:
    rhs = defs.get(ssa)
    if rhs is None:
        return None

    match = re.match(
        r"stablehlo\.constant\s+dense<([^>]+)>\s*:\s*tensor(?:<[^>]*i1[^>]*>)?",
        rhs,
    )
    if match:
        return [bool(v) for v in _parse_dense_int_list(match.group(1))]

    match = re.match(
        r"stablehlo\.compare\s+(LT|GT|EQ|NE|GE|LE),\s+(%[\w.]+),\s+(%[\w.]+)",
        rhs,
    )
    if match:
        op, lhs_ssa, rhs_ssa = match.groups()
        lhs = _eval_i64_tensor(defs, lhs_ssa, memo_i64, visiting)
        rhs_vals = _eval_i64_tensor(defs, rhs_ssa, memo_i64, visiting)
        if lhs is None or rhs_vals is None or len(lhs) != len(rhs_vals):
            return None
        result: list[bool] = []
        for i in range(len(lhs)):
            left, right = lhs[i], rhs_vals[i]
            if op == "LT":
                result.append(left < right)
            elif op == "GT":
                result.append(left > right)
            elif op == "EQ":
                result.append(left == right)
            elif op == "NE":
                result.append(left != right)
            elif op == "GE":
                result.append(left >= right)
            else:
                result.append(left <= right)
        return result

    match = re.match(
        r"stablehlo\.select\s+(%[\w.]+),\s*(%[\w.]+),\s*(%[\w.]+)\s*:",
        rhs,
    )
    if match:
        pred, on_true, on_false = match.groups()
        pred_bits = _eval_i1_tensor(defs, pred, memo_i64, visiting)
        true_bits = _eval_i1_tensor(defs, on_true, memo_i64, visiting)
        false_bits = _eval_i1_tensor(defs, on_false, memo_i64, visiting)
        if pred_bits is None or true_bits is None or false_bits is None:
            return None
        if len(pred_bits) == 1 and len(true_bits) == 1 and len(false_bits) == 1:
            return true_bits if pred_bits[0] else false_bits
        if len(pred_bits) != len(true_bits) or len(pred_bits) != len(false_bits):
            return None
        return [
            true_bits[i] if pred_bits[i] else false_bits[i]
            for i in range(len(pred_bits))
        ]

    return None


def _rewrite_real_dynamic_slice_to_slice(mlir_text: str) -> str:
    """
    Rewrite static stablehlo.real_dynamic_slice to stablehlo.slice.

    onnx-mlir always emits real_dynamic_slice for ONNX Slice; tt-mlir lowers slice only.
    """
    defs = _build_ssa_def_map(mlir_text)
    memo: dict[str, list[int]] = {}

    def _replace(match: re.Match[str]) -> str:
        operand_ty = match.group("in_tys").split(",")[0].strip()
        starts = _eval_i64_tensor(defs, match.group("start"), memo, set())
        limits = _eval_i64_tensor(defs, match.group("limit"), memo, set())
        strides = _eval_i64_tensor(defs, match.group("strides"), memo, set())
        if starts is None or limits is None or strides is None:
            return match.group(0)
        start_s = ", ".join(str(v) for v in starts)
        limit_s = ", ".join(str(v) for v in limits)
        stride_s = ", ".join(str(v) for v in strides)
        return (
            f"{match.group('prefix')}stablehlo.slice {match.group('operand')}, "
            f"[{start_s}], [{limit_s}], [{stride_s}] "
            f": ({operand_ty}) -> {match.group('out_ty')}"
            f"{match.group('suffix')}"
        )

    return _RDS_LINE_RE.sub(_replace, mlir_text)
