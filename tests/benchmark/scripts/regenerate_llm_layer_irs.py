#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regenerate the single-layer LLM TTIR model files consumed by tt-mlir.

Context
-------
tt-mlir ships ~46 checked-in single decoder-layer / prefill-layer TTIR dumps
under ``test/ttmlir/models/single_blocks_and_layers/`` (e.g.
``llama_3_2_1b_decode_layer.mlir``). They are produced by lowering one
transformer layer through the tt-xla (torch-xla + PJRT) stack and dumping the
TTIR stage. When the tt-mlir dialect changes (e.g. the in-place cache ops in
commit d66d735 "Inplace cache ops."), these files must be regenerated so they
match the new IR.

Pipeline
--------
  1. (optional, --run) invoke ``run_one_layer_benchmarks.py`` which runs the
     per-model tests in ``tests/benchmark/test_llms.py`` with ``--num-layers 1``.
     Those tests set the PJRT ``export_path`` so the compiler dumps every stage
     into ``tests/benchmark/modules/irs/`` as
     ``ttir_<model>_1lyr_bs<bs>_isl<isl>_run<id>_g<n>_<ts>.mlir``.
  2. classify each raw TTIR dump by its *cache op content* (robust; the runner's
     own g0/g1 prefill/decode filename labels are unreliable when a model emits
     more than two graphs):
         contains ttir.fill_cache   -> prefill  -> <model>_prefill_layer.mlir
         contains ttir.update_cache -> decode   -> <model>_decode_layer.mlir
     For each (model, kind) the newest dump is chosen. Graphs with neither cache
     op (e.g. encoders) are skipped -- only the cache-bearing layer files are in
     scope for the in-place-cache regeneration.
  3. (optional, --install) run tt-mlir's
     ``tools/scripts/update_llm_perf_tests.sh <staging> --models-only`` to copy
     them into ``test/ttmlir/models/single_blocks_and_layers/``.

IMPORTANT (environments differ)
-------------------------------
Steps 1-2 must run inside the tt-xla environment (``source venv/activate`` in the
tt-xla repo, with tt-xla built against tt-mlir d66d735). Step 3 only shuffles
files; the results should be sanity-checked afterwards with tt-mlir's
``ttmlir-opt`` from the tt-mlir environment.

TP (tensor-parallel, e.g. ``*_tp_*``) and the multi-layer / full-decoder specials
are out of scope here.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# tests/benchmark/scripts/ -> tests/benchmark -> tests -> <tt-xla repo root>
SCRIPTS_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPTS_DIR.parent
TTXLA_ROOT = BENCH_DIR.parent.parent

DEFAULT_IRS_DIR = BENCH_DIR / "modules" / "irs"
DEFAULT_STAGING = TTXLA_ROOT / "transformer_test_irs"

# ttir_<model>_1lyr_bs<bs>_isl<isl>_run<id>_g<n>_<ts>.mlir
_TTIR_RE = re.compile(
    r"^ttir_(?P<model>.+?)_1lyr_bs\d+_isl\d+_run[0-9A-Za-z]+_g(?P<g>\d+)_\d+$"
)

# The dumped model name is the pytest function name minus "test_" (e.g.
# test_falcon3_1b -> falcon3_1b), which does not always match the tt-mlir target
# file stem (falcon_3_1b). Map the mismatched families here. Anything not listed
# is assumed to already match the target stem.
NAME_ALIASES = {
    "falcon3_1b": "falcon_3_1b",
    "falcon3_3b": "falcon_3_3b",
    "phi1": "phi_1",
    "phi1_5": "phi_1_5",
    "phi2": "phi_2",
}


def target_base(model: str) -> str:
    return NAME_ALIASES.get(model, model)


def _default_ttmlir_root() -> Path:
    """Locate the tt-mlir checkout: $TT_MLIR_HOME, else a sibling ../tt-mlir."""
    env = os.environ.get("TT_MLIR_HOME")
    if env:
        return Path(env).resolve()
    return (TTXLA_ROOT.parent / "tt-mlir").resolve()


def run_generator(prefix: list[str], resume: bool, include_tp: bool) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / "run_one_layer_benchmarks.py")]
    if resume:
        cmd.append("--continue")
    if include_tp:
        cmd.append("--include-tp")
    for p in prefix:
        cmd += ["--prefix", p]
    print(f"[regen] running one-layer generator: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def classify(mlir_path: Path) -> str | None:
    """Return 'prefill' (fill_cache), 'decode' (update_cache), or None."""
    text = mlir_path.read_text()
    has_fill = "ttir.fill_cache" in text
    has_update = "ttir.update_cache" in text
    if has_fill and not has_update:
        return "prefill"
    if has_update and not has_fill:
        return "decode"
    if has_fill and has_update:
        # A single layer graph is either prefill or decode, not both. If both
        # appear, prefer decode (update_cache is the decode-defining op).
        return "decode"
    return None


def collect(irs_dir: Path, prefix: list[str]) -> dict[tuple[str, str], Path]:
    """Map (model, kind) -> chosen (newest) raw TTIR dump."""
    best: dict[tuple[str, str], Path] = {}
    if not irs_dir.exists():
        print(f"[regen] ERROR: irs dir does not exist: {irs_dir}", file=sys.stderr)
        return best
    for mlir in irs_dir.glob("ttir_*.mlir"):
        m = _TTIR_RE.match(mlir.stem)
        if not m:
            continue
        model = m.group("model")
        if model.endswith("_tp"):
            continue  # TP handled separately
        if prefix and not any(model.startswith(p) for p in prefix):
            continue
        kind = classify(mlir)
        if kind is None:
            continue
        key = (model, kind)
        cur = best.get(key)
        if cur is None or mlir.stat().st_mtime > cur.stat().st_mtime:
            best[key] = mlir
    return best


def stage(
    chosen: dict[tuple[str, str], Path],
    staging: Path,
    existing_targets: set[str] | None,
) -> list[str]:
    """Copy chosen dumps into staging using tt-mlir target names.

    If existing_targets is provided, only stage files that correspond to an
    already-existing tt-mlir target (so we overwrite the checked-in 43 files and
    never introduce spurious new model files); others are reported as skipped.
    """
    staging.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    skipped: list[str] = []
    for (model, kind), src in sorted(chosen.items()):
        target = f"{target_base(model)}_{kind}_layer"
        if existing_targets is not None and target not in existing_targets:
            skipped.append(f"{target} (no such tt-mlir target; from {src.name})")
            continue
        dest = staging / f"{target}.mlir"
        shutil.copy2(src, dest)
        names.append(target)
        print(f"[regen] {src.name}  ->  {dest.name}")
    for s in skipped:
        print(f"[regen] SKIP {s}")
    return names


def existing_target_stems(ttmlir_root: Path) -> set[str]:
    models_dir = ttmlir_root / "test" / "ttmlir" / "models" / "single_blocks_and_layers"
    return {p.stem for p in models_dir.glob("*.mlir")}


def install(staging: Path, ttmlir_root: Path, models: list[str]) -> None:
    script = ttmlir_root / "tools" / "scripts" / "update_llm_perf_tests.sh"
    if not script.exists():
        print(f"[regen] ERROR: installer not found: {script}", file=sys.stderr)
        sys.exit(1)
    try:
        src_arg = str(staging.resolve().relative_to(ttmlir_root))
    except ValueError:
        src_arg = str(staging.resolve())
    args = [str(script), src_arg]
    args += models if models else ["--models-only"]
    print(f"[regen] installing into tt-mlir: {' '.join(args)}")
    subprocess.run(args, check=True, cwd=str(ttmlir_root))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", action="store_true",
                    help="First run the one-layer benchmark generator (requires a TT device).")
    ap.add_argument("--continue", dest="resume", action="store_true",
                    help="Pass --continue to the generator (skip already-produced models).")
    ap.add_argument("--include-tp", action="store_true",
                    help="Also run the tensor-parallel (_tp) generator tests (not staged here).")
    ap.add_argument("--prefix", action="append", default=[],
                    help="Only (re)generate models whose name starts with this prefix. Repeatable.")
    ap.add_argument("--irs-dir", type=Path, default=DEFAULT_IRS_DIR,
                    help=f"Directory of raw PJRT IR dumps (default: {DEFAULT_IRS_DIR}).")
    ap.add_argument("--staging", type=Path, default=DEFAULT_STAGING,
                    help=f"Staging dir with tt-mlir-named files (default: {DEFAULT_STAGING}).")
    ap.add_argument("--install", action="store_true",
                    help="After staging, install into tt-mlir via update_llm_perf_tests.sh.")
    ap.add_argument("--ttmlir-root", type=Path, default=None,
                    help="tt-mlir checkout (default: $TT_MLIR_HOME or ../tt-mlir).")
    ap.add_argument("--no-target-filter", action="store_true",
                    help="Stage every classified model, even without an existing tt-mlir target.")
    ap.add_argument("--models", nargs="*", default=[],
                    help="Specific model names to install (default: --models-only for all staged).")
    args = ap.parse_args()

    if args.run:
        run_generator(args.prefix, args.resume, args.include_tp)

    ttmlir_root = (args.ttmlir_root or _default_ttmlir_root()).resolve()
    existing = None if args.no_target_filter else existing_target_stems(ttmlir_root)

    chosen = collect(args.irs_dir, args.prefix)
    names = stage(chosen, args.staging, existing)
    prefill = sum(1 for n in names if n.endswith("_prefill_layer"))
    decode = sum(1 for n in names if n.endswith("_decode_layer"))
    print(f"\n[regen] staged {len(names)} file(s) ({prefill} prefill, {decode} decode) "
          f"in {args.staging}")

    if args.install:
        install(args.staging, ttmlir_root, args.models)
        print(f"[regen] installed into {ttmlir_root}/test/ttmlir/models/single_blocks_and_layers")

    return 0 if names else 1


if __name__ == "__main__":
    raise SystemExit(main())
