# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers to summarize GroupNorm-related patterns in exported MLIR."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Playground-style composite / fused GN path (good).
COMPOSITE_MARKERS = (
    "tenstorrent.group_norm",
    "composite_group_norm",
    "ttir.group_norm",
    "ttnn.group_norm",
)

# Decomposed GroupNorm — tile-padded subtract OOM class (bad).
DECOMPOSED_MARKERS = (
    "stablehlo.subtract",
    "ttnn.subtract",
    "ttir.subtract",
    "binary_ng",
)


def _latest_mlir_files(export_dir: Path, stage: str) -> list[Path]:
    irs = export_dir / "irs"
    if not irs.is_dir():
        return []
    return sorted(irs.glob(f"{stage}_*.mlir"), key=lambda p: p.stat().st_mtime)


def _count_markers(text: str, markers: Iterable[str]) -> dict[str, int]:
    return {m: text.count(m) for m in markers}


def summarize_export_dir(export_dir: Path) -> dict:
    """Return per-stage marker counts for the newest MLIR files under export_dir/irs."""
    summary: dict = {"export_dir": str(export_dir), "stages": {}}
    for stage in ("vhlo", "shlo", "ttir", "ttnn"):
        files = _latest_mlir_files(export_dir, stage)
        if not files:
            summary["stages"][stage] = {"file": None, "composite": {}, "decomposed": {}}
            continue
        path = files[-1]
        text = path.read_text(errors="replace")
        summary["stages"][stage] = {
            "file": path.name,
            "composite": _count_markers(text, COMPOSITE_MARKERS),
            "decomposed": _count_markers(text, DECOMPOSED_MARKERS),
        }
    return summary


def format_summary(summary: dict) -> str:
    lines = [f"IR export: {summary['export_dir']}"]
    for stage, info in summary["stages"].items():
        fname = info.get("file") or "(missing)"
        comp = {k: v for k, v in info.get("composite", {}).items() if v}
        decomp = {k: v for k, v in info.get("decomposed", {}).items() if v}
        lines.append(f"  {stage:4s} {fname}")
        lines.append(f"         composite: {comp or 'none'}")
        lines.append(f"         decomposed: {decomp or 'none'}")
    return "\n".join(lines)
