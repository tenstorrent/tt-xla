# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fusion-pattern verification for benchmark tests.

Accepts a list of expected op strings (e.g. ["ttnn.rms_norm", "ttnn.scaled_dot_product_attention"]),
globs the IR files produced by the current run, and searches for each op by simple
string match. A pattern must appear in every graph file.

IR type is inferred from the op prefix e.g. "ttnn." -> ttnn, "ttir." -> ttir.
The current run is identified by `export_model_name`, which embeds a unique
`_run<hex>_` token, so files from previous runs are naturally excluded.
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
) -> list[Path]:
    """
    Return graph-IR files produced by the current benchmark run.

    Globs `{modules_dir}/irs/{ir_type}_{export_model_name}_*.mlir`. The
    `export_model_name` carries a unique `_run<hex>_` token so the match is
    already scoped to the current run. Sorted for deterministic iteration.
    """
    irs_dir = Path(modules_dir) / "irs"
    if not irs_dir.is_dir():
        return []

    prefix = f"{ir_type}_{export_model_name}_"
    return sorted(irs_dir.glob(f"{prefix}*.mlir"))


def check_fusions(
    expected_ops: list[str],
    export_model_name: str,
    modules_dir: str | Path,
) -> None:
    """
    Verify that expected ops appear in the IR produced by this benchmark run.

    For each op in expected_ops the IR type is inferred from the op's prefix
    (e.g. "ttnn." -> ttnn IR files). The op must be present in every matching
    graph file.

    Args:
        expected_ops: List of op strings like "ttnn.rms_norm".
        export_model_name: Model export name used to glob IR file names.
            Already unique per run via the `_run<hex>_` token it embeds.
        modules_dir: Directory containing the "irs/" subdirectory.

    Raises:
        AssertionError: if any op is absent from any graph file for this run,
            or if no IR files can be found.
    """
    if not expected_ops:
        return

    failures: list[str] = []

    for op in expected_ops:
        ir_type = _infer_ir_type(op)

        ir_files = _collect_ir_files(
            modules_dir=modules_dir,
            ir_type=ir_type,
            export_model_name=export_model_name,
        )
        if not ir_files:
            failures.append(
                f"Op '{op}': no IR files found "
                f"(looked for {ir_type}_{export_model_name}_*.mlir under "
                f"{modules_dir}/irs/)"
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
