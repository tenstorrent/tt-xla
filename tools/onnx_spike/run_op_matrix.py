#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WS4 op matrix runner (onnx-mlir bridge only).

Prerequisites:
  source venv/activate
  source tools/onnx/env.sh
  pip install onnx numpy onnxruntime

Usage:
  python tools/onnx_spike/run_op_matrix.py
  python tools/onnx_spike/run_op_matrix.py --full
  python tools/onnx_spike/run_op_matrix.py --full --report tools/onnx/build/tt_onnx/op_matrix/full_report.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python_package") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python_package"))

from tt_onnx.op_matrix import FULL_OPS, SEED_OPS, run_op_matrix  # noqa: E402
from tt_onnx.plugin_check import verify_pjrt_plugin_loads  # noqa: E402


def _load_gen_op_onnx():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_op_onnx", REPO_ROOT / "tools" / "onnx" / "gen_op_onnx.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_fixtures(fixture_dir: Path, ops: tuple[str, ...], *, full: bool) -> None:
    missing = [op for op in ops if not (fixture_dir / f"{op}.onnx").is_file()]
    if not missing:
        return
    gen = REPO_ROOT / "tools" / "onnx" / "gen_op_onnx.py"
    cmd = [sys.executable, str(gen), "--all"]
    if full:
        cmd.append("--full")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run M1.8 full op matrix (17 ops) instead of 5 seed ops",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=REPO_ROOT / "tools" / "onnx" / "fixtures",
        help="Directory containing <op>.onnx fixtures",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Per-op bridge/compile artifact root (default depends on --full)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report output path (default depends on --full)",
    )
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    ops = FULL_OPS if args.full else SEED_OPS
    label = "full" if args.full else "seed"
    default_work = REPO_ROOT / "tools" / "onnx" / "build" / "tt_onnx" / f"op_matrix_{label}"
    default_report = default_work / "report.json"
    work_root = args.work_root or default_work
    report_path = args.report or default_report

    print("==> Checking PJRT plugin ...")
    verify_pjrt_plugin_loads()

    print(f"==> Ensuring {label} op fixtures ({len(ops)} ops) ...")
    _ensure_fixtures(args.fixture_dir, ops, full=args.full)
    gen_mod = _load_gen_op_onnx()
    feeds = {op: gen_mod.default_feed(op) for op in ops}

    cases = [(op, args.fixture_dir / f"{op}.onnx", feeds[op]) for op in ops]

    print(f"==> Running {label} op matrix ({len(cases)} ops) ...")
    report = run_op_matrix(
        cases,
        work_root=work_root,
        tolerance=args.tolerance,
        compile_options={"optimization_level": "0"},
        matrix=label,
        ops=ops,
    )
    report.write_json(report_path)

    for result in report.results:
        status = result.status.value
        diff = "" if result.max_abs_diff is None else f" diff={result.max_abs_diff:.6g}"
        err = "" if not result.error else f" ({result.error})"
        print(f"    {result.op:12s} {status}{diff}{err}")

    pass_rate = 100.0 * report.pass_count / len(report.results) if report.results else 0.0
    print(f"==> Report: {report_path}")
    print(f"    pass={report.pass_count} fail={report.fail_count} ({pass_rate:.0f}%)")

    if report.fail_count:
        return 1
    print(f"PASSED: {label} op matrix on TT device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
