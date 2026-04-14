#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Rename legacy long ``export_model_name`` MLIR/TTNN files to short ``{stage}_{dec|all}_gN_{ts}`` names.

Scans ``tests/benchmark/modules/gpt_oss_input_sharding_dbg`` by default. Safe to re-run;
skips files that do not match the legacy pattern or if the target name already exists.

    cd tests/benchmark
    python scripts/rename_gpt_oss_sharding_export_files.py
    python scripts/rename_gpt_oss_sharding_export_files.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Longest first (prefix match).
_STAGES = sorted(
    (
        "shlo_compiler_cleaned",
        "shlo_set_mesh_attr",
        "shlo_frontend",
        "shlo_compiler",
        "shlo",
        "vhlo",
        "ttir",
        "ttnn",
    ),
    key=len,
    reverse=True,
)


def _parse_stage(filename_base: str) -> tuple[str, str] | tuple[None, None]:
    for st in _STAGES:
        p = st + "_"
        if filename_base.startswith(p):
            return st, filename_base[len(p) :]
    return None, None


def _mode_and_g_from_body(body: str) -> tuple[str, str] | tuple[None, None]:
    mg = re.search(r"_g(\d+)$", body)
    if not mg:
        return None, None
    g = mg.group(1)
    if "ir_actbatchshard_skip_prefill" in body:
        return "dec", g
    if "ir_actbatchshard_prefill_decode" in body:
        return "all", g
    return None, None


def _new_mlir_name(name: str) -> str | None:
    if not name.endswith(".mlir"):
        return None
    base = name[:-5]
    st, rem = _parse_stage(base)
    if not st or not rem:
        return None
    body, _, ts = rem.rpartition("_")
    if not body or not ts.isdigit():
        return None
    mode, g = _mode_and_g_from_body(body)
    if mode is None:
        return None
    return f"{st}_{mode}_g{g}_{ts}.mlir"


def _new_fb_name(name: str) -> str | None:
    if not (name.startswith("fb_") and name.endswith(".ttnn")):
        return None
    inner = name[3:-5]
    body, _, ts = inner.rpartition("_")
    if not body or not ts.isdigit():
        return None
    mode, g = _mode_and_g_from_body(body)
    if mode is None:
        return None
    return f"fb_{mode}_g{g}_{ts}.ttnn"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "modules" / "gpt_oss_input_sharding_dbg",
        help="Export root (contains irs/ and fb_*.ttnn)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print planned renames only")
    args = ap.parse_args()
    root: Path = args.root
    irs = root / "irs"
    if not irs.is_dir():
        print(f"No directory {irs}", file=sys.stderr)
        raise SystemExit(1)

    pairs: list[tuple[Path, Path]] = []
    for p in sorted(irs.glob("*.mlir")):
        nn = _new_mlir_name(p.name)
        if nn is None or nn == p.name:
            continue
        pairs.append((p, p.with_name(nn)))
    for p in sorted(root.glob("fb_*.ttnn")):
        nn = _new_fb_name(p.name)
        if nn is None or nn == p.name:
            continue
        pairs.append((p, p.with_name(nn)))

    for src, dst in pairs:
        if dst.exists():
            print(f"SKIP exists: {dst.name}", file=sys.stderr)
            continue
        print(f"{src.relative_to(root)} -> {dst.relative_to(root)}")
        if not args.dry_run:
            src.rename(dst)

    if not pairs:
        print("No legacy files matched (nothing to rename).")


if __name__ == "__main__":
    main()
