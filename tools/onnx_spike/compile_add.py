#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WS3 spike: compile and run the trivial Add ONNX model on a TT device.

Prerequisites:
  source venv/activate
  source tools/onnx/env.sh
  pip install onnx numpy onnxruntime

Usage (from tt-xla repo root):
  python tools/onnx_spike/compile_add.py
  python tools/onnx_spike/compile_add.py --onnx path/to/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python_package") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python_package"))

from tt_onnx.plugin_check import verify_pjrt_plugin_loads  # noqa: E402


def _default_add_onnx() -> Path:
    fixture = REPO_ROOT / "tools" / "onnx" / "fixtures" / "add.onnx"
    if fixture.is_file():
        return fixture
    gen_script = REPO_ROOT / "tools" / "onnx" / "gen_add_onnx.py"
    if gen_script.is_file():
        import subprocess

        subprocess.run(
            [sys.executable, str(gen_script), "-o", str(fixture)],
            check=True,
        )
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=_default_add_onnx(),
        help="Path to Add ONNX model",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "tools" / "onnx" / "build" / "tt_onnx" / "add_spike",
        help="Bridge + export artifact directory",
    )
    args = parser.parse_args()

    a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    b = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    feed = {"A": a, "B": b}

    print("==> Checking PJRT plugin ...")
    verify_pjrt_plugin_loads()

    from tt_onnx import ONNXSession  # noqa: E402
    from tt_onnx.runtime import max_abs_diff, run_onnxruntime_cpu  # noqa: E402

    print(f"==> Loading and compiling {args.onnx} ...")
    session = ONNXSession(
        args.onnx,
        work_dir=args.work_dir,
        compile_options={"optimization_level": "0"},
    )
    print(f"    Bridge ONNX dialect: {session.bridge_artifacts.onnx_dialect_mlir}")
    print(f"    Bridge StableHLO:    {session.bridge_artifacts.stablehlo_mlir}")
    print(f"    Compile time:        {session.compile_artifacts.compile_time_s:.2f}s")

    print("==> Running on TT device ...")
    tt_out = session.run(feed)
    ref_out = run_onnxruntime_cpu(str(args.onnx), feed)

    print("==> Comparing vs ONNX Runtime CPU ...")
    for name in session.output_names:
        diff = max_abs_diff(tt_out[name], ref_out[name])
        print(f"    {name}: max_abs_diff={diff:.6g}")
        if diff > 1e-3:
            print("FAILED: numerical mismatch", file=sys.stderr)
            return 1

    print("PASSED: Add ONNX e2e on TT device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
