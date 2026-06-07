#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Triage one op from the M1.8 matrix: bridge → canonicalize → compile → execute.

Run on the reservation host (venv + tools/onnx/env.sh):

  python tools/onnx_spike/triage_op.py reduce_mean
  python tools/onnx_spike/triage_op.py conv --bridge-only
  python tools/onnx_spike/triage_op.py gather --compile-only

Artifacts land under tools/onnx/build/tt_onnx/triage/<op>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python_package") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python_package"))

from tt_onnx.bridge import OnnxBridge  # noqa: E402
from tt_onnx.mlir_utils import canonicalize_onnx_stablehlo  # noqa: E402
from tt_onnx.op_matrix import FULL_OPS, OpMatrixStatus, op_tolerance, run_op_case  # noqa: E402


def _load_gen_op_onnx():
    spec = importlib.util.spec_from_file_location(
        "gen_op_onnx", REPO_ROOT / "tools" / "onnx" / "gen_op_onnx.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_fixture(op: str, fixture_dir: Path) -> Path:
    onnx_path = fixture_dir / f"{op}.onnx"
    if not onnx_path.is_file():
        gen = REPO_ROOT / "tools" / "onnx" / "gen_op_onnx.py"
        subprocess.run(
            [sys.executable, str(gen), op],
            check=True,
        )
    return onnx_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", choices=FULL_OPS, help="Op matrix case to triage")
    parser.add_argument(
        "--bridge-only",
        action="store_true",
        help="Stop after onnx-mlir + canonicalize (no PJRT)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile on TT but skip execute/compare",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=REPO_ROOT / "tools" / "onnx" / "fixtures",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Artifact root (default: tools/onnx/build/tt_onnx/triage/<op>)",
    )
    args = parser.parse_args()

    work_dir = args.work_dir or (
        REPO_ROOT / "tools" / "onnx" / "build" / "tt_onnx" / "triage" / args.op
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    gen_mod = _load_gen_op_onnx()
    onnx_path = _ensure_fixture(args.op, args.fixture_dir)
    feed = gen_mod.default_feed(args.op)

    print(f"==> Op: {args.op}")
    print(f"    ONNX: {onnx_path}")

    print("==> Bridge (onnx-mlir → StableHLO) ...")
    try:
        bridge = OnnxBridge()
        artifacts = bridge.convert(onnx_path, work_dir)
    except Exception:
        print("FAILED at bridge (ONNX→SHLO):", file=sys.stderr)
        traceback.print_exc()
        return 1

    raw = artifacts.stablehlo_text
    raw_path = work_dir / f"{args.op}.stablehlo.raw.mlir"
    raw_path.write_text(raw, encoding="utf-8")

    canonical = canonicalize_onnx_stablehlo(raw)
    canon_path = work_dir / f"{args.op}.stablehlo.canonical.mlir"
    canon_path.write_text(canonical, encoding="utf-8")

    print(f"    raw:        {raw_path}")
    print(f"    canonical:  {canon_path}")

    for label, text in (("raw", raw), ("canonical", canonical)):
        flags = []
        if "shape." in text:
            flags.append("shape-dialect")
        if "onnx.NoValue" in text or "NoValue" in text:
            flags.append("NoValue")
        if "stablehlo.dot " in text:
            flags.append("stablehlo.dot")
        if "real_dynamic_slice" in text:
            flags.append("real_dynamic_slice")
        if "custom_call" in text or "torch_index_select" in text:
            flags.append("custom_call")
        if flags:
            print(f"    {label} flags: {', '.join(flags)}")

    if args.bridge_only:
        print("==> bridge-only: done")
        return 0

    if args.compile_only:
        from tt_onnx.compiler import compile_stablehlo_mlir, get_tt_device  # noqa: E402
        from tt_onnx.plugin_check import verify_pjrt_plugin_loads  # noqa: E402

        print("==> PJRT compile-only ...")
        verify_pjrt_plugin_loads()
        try:
            _, compile_art = compile_stablehlo_mlir(
                raw,
                {
                    "mlir_input_format": "auto",
                    "export_path": str(work_dir / "export"),
                    "export_model_name": args.op,
                    "optimization_level": "0",
                },
                device=get_tt_device(),
            )
        except Exception:
            print("FAILED at compile (SHLO→TTNN):", file=sys.stderr)
            traceback.print_exc()
            ir_dir = work_dir / "export" / "irs"
            if ir_dir.is_dir():
                dumps = sorted(ir_dir.glob("*.mlir"))
                if dumps:
                    print(f"    IR dumps: {ir_dir}", file=sys.stderr)
                    for p in dumps[-3:]:
                        print(f"      {p.name}", file=sys.stderr)
            return 1
        print(f"    compile_time_s={compile_art.compile_time_s:.3f}")
        print("==> compile-only: PASSED")
        return 0

    from tt_onnx.plugin_check import verify_pjrt_plugin_loads  # noqa: E402

    print("==> Full e2e (compile + execute vs ORT) ...")
    verify_pjrt_plugin_loads()
    result = run_op_case(
        args.op,
        onnx_path,
        feed,
        work_dir=work_dir / "session",
        tolerance=op_tolerance(args.op),
        compile_options={"optimization_level": "0"},
    )
    print(f"    status={result.status.value}")
    if result.max_abs_diff is not None:
        print(f"    max_abs_diff={result.max_abs_diff:.6g}")
    if result.error:
        print(f"    error={result.error}")
    if result.status == OpMatrixStatus.COMPILE_FAIL:
        print("    stage=ONNX→SHLO or SHLO→TTNN (check export/irs/)", file=sys.stderr)
    elif result.status == OpMatrixStatus.EXECUTE_FAIL:
        print("    stage=execute", file=sys.stderr)
    elif result.status == OpMatrixStatus.NUMERICAL_FAIL:
        print("    stage=numerical", file=sys.stderr)

    return 0 if result.status == OpMatrixStatus.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
