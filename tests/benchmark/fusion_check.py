# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fusion-pattern verification for benchmark tests.

Accepts a list of OpCount objects that specify expected per-layer op counts,
globs the ttnn IR files produced by the current run, and verifies exact
occurrence counts (per_layer_count * num_layers) in each graph file.

The current run is identified by `export_model_name`, which embeds a unique
`_run<hex>_` token, so files from previous runs are naturally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpCountPerLayer:
    """Expected per-layer count for a TTNN op.

    The total expected occurrences in an IR file is ``count * num_layers``.
    """

    op: str
    count: int


@dataclass
class OpCountPerLayerBetween:
    """Expected per-layer count range for a TTNN op.

    The total expected occurrences must be between
    ``min_count * num_layers`` and ``max_count * num_layers`` (inclusive).
    """

    op: str
    min_count: int
    max_count: int


FusionCheck = OpCountPerLayer | OpCountPerLayerBetween

# Ops that are commonly fused in TTNN IR. When no expected_ops are specified,
# these are counted and printed for informational purposes.
DISCOVERY_OPS = [
    "ttnn.rotary_embedding",
    "ttnn.scaled_dot_product_attention",
    "ttnn.scaled_dot_product_attention_decode",
]


def _load_ir_files(
    export_model_name: str, modules_dir: str | Path
) -> list[tuple[Path, str]]:
    irs_dir = Path(modules_dir) / "irs"
    prefix = f"ttnn_{export_model_name}_g"
    ir_files = sorted(irs_dir.glob(f"{prefix}*.mlir")) if irs_dir.is_dir() else []
    return [(f, f.read_text()) for f in ir_files]


def _count_op(ir_contents: list[tuple[Path, str]], op: str) -> int:
    op_pattern = f'"{op}"'
    return sum(text.count(op_pattern) for _, text in ir_contents)


def check_fusions(
    expected_ops: list[FusionCheck],
    export_model_name: str,
    modules_dir: str | Path,
    num_layers: int,
    strict: bool = False,
) -> None:
    """
    Verify that expected ops appear the correct number of times in the ttnn IR.

    Always prints fusion stats. When ``strict=True``, raises on mismatches.

    Args:
        expected_ops: List of FusionCheck objects.
        export_model_name: Model export name used to glob IR file names.
        modules_dir: Directory containing the "irs/" subdirectory.
        num_layers: Number of transformer layers in the model.
        strict: If True, raise on mismatches. If False, only print stats.

    Raises:
        AssertionError: if strict=True and any check fails or no IR files are found.
    """
    ir_contents = _load_ir_files(export_model_name, modules_dir)
    if not ir_contents:
        msg = (
            f"Fusion check: no IR files found "
            f"(looked for ttnn_{export_model_name}_g*.mlir under {modules_dir}/irs/)"
        )
        if strict:
            raise AssertionError(msg)
        print(f"WARNING: {msg}")
        return

    checked_ops = {check.op for check in expected_ops} if expected_ops else set()

    # Print discovery stats for common fused ops not covered by explicit checks
    uncovered = [op for op in DISCOVERY_OPS if op not in checked_ops]
    if uncovered:
        print(f"\nFusion discovery ({len(ir_contents)} IR files, {num_layers} layers):")
        for op in uncovered:
            count = _count_op(ir_contents, op)
            if count > 0:
                print(
                    f"  INFO: '{op}' count={count} ({count / num_layers:.1f} per layer)"
                )

    if not expected_ops:
        return

    # Validate explicit checks
    print(f"\nFusion check ({len(ir_contents)} IR files, {num_layers} layers):")
    failures: list[str] = []
    for check in expected_ops:
        actual_count = _count_op(ir_contents, check.op)

        if isinstance(check, OpCountPerLayerBetween):
            min_expected = check.min_count * num_layers
            max_expected = check.max_count * num_layers
            passed = min_expected <= actual_count <= max_expected
            status = "PASS" if passed else "FAIL"
            msg = (
                f"  {status}: '{check.op}' count={actual_count}, "
                f"expected [{min_expected}..{max_expected}] "
                f"([{check.min_count}..{check.max_count}] * {num_layers} layers)"
            )
            print(msg)
            if not passed:
                failures.append(msg.strip())
        else:
            expected_count = check.count * num_layers
            passed = actual_count == expected_count
            status = "PASS" if passed else "FAIL"
            msg = (
                f"  {status}: '{check.op}' count={actual_count}, "
                f"expected {expected_count} "
                f"({check.count} * {num_layers} layers)"
            )
            print(msg)
            if not passed:
                failures.append(msg.strip())

    if failures and strict:
        raise AssertionError("Fusion check failed:\n" + "\n".join(failures))
