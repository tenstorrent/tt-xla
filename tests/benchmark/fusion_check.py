# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fusion-pattern verification for benchmark tests.

Accepts a list of expected op strings (e.g. ["ttnn.rms_norm", "ttnn.scaled_dot_product_attention"]),
globs the IR files produced by the current run, and searches for each op by simple
string match. A pattern must appear in every graph file.

IR type is inferred from the op prefix e.g. "ttnn." -> ttnn, "ttir." -> ttir.
"""

from __future__ import annotations

from pathlib import Path


def _infer_ir_type(op: str) -> str:
    """Derive IR stage from op name prefix (e.g. "ttnn.rms_norm" -> "ttnn")."""
    return op.split(".")[0] if "." in op else "ttnn"


def _collect_ir_files(
    modules_dir: str | Path,
    ir_type: str,
    export_model_name: str,
    min_mtime: float,
) -> list[Path]:
    """
    Return graph-IR files produced by the current benchmark run.

    Globs `{modules_dir}/irs/{ir_type}_{export_model_name}_*.mlir` and
    filters to files whose mtime is >= min_mtime (to exclude stale files
    from previous runs). Sorted for deterministic iteration.
    """
    irs_dir = Path(modules_dir) / "irs"
    if not irs_dir.is_dir():
        return []

    prefix = f"{ir_type}_{export_model_name}_"
    return sorted(
        p for p in irs_dir.glob(f"{prefix}*.mlir") if p.stat().st_mtime >= min_mtime
    )


def check_fusions(
    expected_ops: list[str | tuple[str, int]],
    export_model_name: str,
    modules_dir: str | Path,
    before_compile_ts: float,
) -> None:
    """
    Verify that expected ops appear in the IR produced by this benchmark run.

    For each op in expected_ops the IR type is inferred from the op's prefix
    (e.g. "ttnn." -> ttnn IR files). The op must be present in every matching
    graph file.

    Args:
        expected_ops: List of op strings like "ttnn.rms_norm".
        export_model_name: Model export name used to glob IR file names.
        modules_dir: Directory containing the "irs/" subdirectory.
        before_compile_ts: Unix timestamp recorded before compilation; only
            IR files newer than this are considered to belong to the current run.

    Raises:
        AssertionError: if any op is absent from any graph file for this run,
            or if no IR files can be found.
    """
    if not expected_ops:
        return

    failures: list[str] = []

    for entry in expected_ops:
        op = entry[0] if isinstance(entry, tuple) else entry
        ir_type = _infer_ir_type(op)

        ir_files = _collect_ir_files(
            modules_dir=modules_dir,
            ir_type=ir_type,
            export_model_name=export_model_name,
            min_mtime=before_compile_ts,
        )
        if not ir_files:
            failures.append(
                f"Op '{op}': no IR files found "
                f"(looked for {ir_type}_{export_model_name}_*.mlir under "
                f"{modules_dir}/irs/ with mtime >= {before_compile_ts})"
            )
            continue

        missing_in = [f for f in ir_files if op not in f.read_text()]
        if missing_in:
            failures.append(
                f"Op '{op}' missing in {len(missing_in)}/{len(ir_files)} "
                f"{ir_type} graph file(s): {[f.name for f in missing_in]}"
            )

    if failures:
        raise AssertionError("Fusion check failed:\n" + "\n".join(failures))
