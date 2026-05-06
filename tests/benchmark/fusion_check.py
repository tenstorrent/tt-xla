# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fusion-pattern verification for benchmark tests.

Accepts a list of expected op strings (e.g. ["ttnn.rms_norm", "ttnn.scaled_dot_product_attention"]),
globs the ttnn IR files produced by the current run, and searches for each op
by simple string match. A pattern must appear in every graph file.

The current run is identified by `export_model_name`, which embeds a unique
`_run<hex>_` token, so files from previous runs are naturally excluded.
"""

from __future__ import annotations

from pathlib import Path


def check_fusions(
    expected_ops: list[str],
    export_model_name: str,
    modules_dir: str | Path,
) -> None:
    """
    Verify that expected ops appear in the ttnn IR produced by this benchmark run.

    Each op must be present in every matching ttnn graph file.

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

    irs_dir = Path(modules_dir) / "irs"
    prefix = f"ttnn_{export_model_name}_"
    ir_files = sorted(irs_dir.glob(f"{prefix}*.mlir")) if irs_dir.is_dir() else []

    if not ir_files:
        raise AssertionError(
            f"Fusion check failed: no IR files found "
            f"(looked for {prefix}*.mlir under {modules_dir}/irs/)"
        )

    ir_contents = [(f, f.read_text()) for f in ir_files]

    failures: list[str] = []
    for op in expected_ops:
        missing_in = [f.name for f, text in ir_contents if op not in text]
        if missing_in:
            failures.append(
                f"Op '{op}' missing in {len(missing_in)}/{len(ir_files)} "
                f"ttnn graph file(s): {missing_in}"
            )

    if failures:
        raise AssertionError("Fusion check failed:\n" + "\n".join(failures))
