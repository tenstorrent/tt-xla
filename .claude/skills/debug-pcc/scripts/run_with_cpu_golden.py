#!/usr/bin/env python3

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Run generated TTNN codegen with CPU golden comparison enabled.

Uses TTNN's built-in ``enable_comparison_mode`` which automatically computes
a CPU reference (via each op's ``golden_function``) for every TTNN operation
and reports PCC between device output and CPU golden.

This is the simplest way to check whether device execution diverges from
CPU truth -- no custom instrumentation needed.

IMPORTANT: TTNN must be imported with ``enable_fast_runtime_mode=false``
for the comparison decorators to be active. This script sets the
``TTNN_CONFIG_OVERRIDES`` env var BEFORE importing ttnn to ensure ops are
registered with the full Operation class (not FastOperation).

USAGE:
------

1. Generate codegen with 1 decoder layer (fast iteration):

   # In llm_benchmark.py, ensure "backend": "codegen_py" and "export_tensors": True
   pytest -svv tests/benchmark/test_llms.py::test_gpt_oss_20b_tp --num-layers 1

2. Run the generated codegen with CPU golden comparison:

   # Log all PCC mismatches (does not stop on failure):
   python scripts/run_with_cpu_golden.py modules/main.py

   # Stop on first PCC failure:
   python scripts/run_with_cpu_golden.py modules/main.py --raise-on-failure

   # Run the vanilla version instead:
   python scripts/run_with_cpu_golden.py modules/main_vanilla.py

   # Custom PCC threshold (default 0.999):
   python scripts/run_with_cpu_golden.py modules/main.py --pcc 0.99

3. Compare vanilla vs optimized CPU golden profiles:

   # Run vanilla first, then optimized -- compare the logs to see which
   # ops diverge from CPU truth in each version.
   python scripts/run_with_cpu_golden.py modules/main_vanilla.py 2>&1 | tee vanilla_cpu_golden.log
   python scripts/run_with_cpu_golden.py modules/main.py 2>&1 | tee optimized_cpu_golden.log

PREREQUISITES:
--------------

  - TT_MLIR_HOME environment variable must be set
  - The tt-xla virtual environment must be activated (source venv/activate)
  - Generated modules/main.py and modules/tensors/ must exist
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_ttnn_env():
    """Set up TT_METAL_RUNTIME_ROOT and sys.path for TTNN imports."""
    tt_mlir_home = os.environ.get("TT_MLIR_HOME")
    if not tt_mlir_home:
        print("ERROR: TT_MLIR_HOME environment variable is not set.")
        sys.exit(1)

    tt_metal_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if not tt_metal_root:
        tt_metal_root = os.path.join(
            tt_mlir_home, "third_party", "tt-metal", "src", "tt-metal"
        )
        os.environ["TT_METAL_RUNTIME_ROOT"] = tt_metal_root
        print(f"TT_METAL_RUNTIME_ROOT set to {tt_metal_root}")

    for subdir in ("ttnn", "tools"):
        p = os.path.join(tt_metal_root, subdir)
        if p not in sys.path:
            sys.path.insert(0, p)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run generated TTNN codegen with CPU golden comparison enabled. "
            "TTNN's built-in comparison mode computes a CPU reference for every "
            "op and reports PCC between device and CPU outputs."
        ),
    )
    parser.add_argument(
        "main_py",
        help="Path to the generated main.py (or main_vanilla.py)",
    )
    parser.add_argument(
        "--pcc",
        type=float,
        default=0.999,
        help="PCC threshold for comparison (default: 0.999)",
    )
    parser.add_argument(
        "--raise-on-failure",
        action="store_true",
        help="Stop execution on first PCC failure (default: log and continue)",
    )
    args = parser.parse_args()

    _ensure_ttnn_env()

    # CRITICAL: Set TTNN_CONFIG_OVERRIDES BEFORE importing ttnn.
    #
    # TTNN registers ops at import time using either FastOperation (default,
    # enable_fast_runtime_mode=true) or Operation (full instrumentation).
    # FastOperation.__call__ skips comparison mode entirely.
    # We must disable fast_runtime_mode before import so ops use Operation.
    os.environ["TTNN_CONFIG_OVERRIDES"] = json.dumps({
        "enable_fast_runtime_mode": False,
        "enable_comparison_mode": True,
        "comparison_mode_pcc": args.pcc,
        "comparison_mode_should_raise_exception": args.raise_on_failure,
    })

    import importlib.util

    import ttnn

    print(f"CPU golden comparison enabled:")
    print(f"  fast_runtime_mode: {ttnn.CONFIG.enable_fast_runtime_mode}")
    print(f"  comparison_mode: {ttnn.CONFIG.enable_comparison_mode}")
    print(f"  PCC threshold: {ttnn.CONFIG.comparison_mode_pcc}")
    print(f"  Raise on failure: {ttnn.CONFIG.comparison_mode_should_raise_exception}")
    print(f"  Codegen: {args.main_py}")

    main_py_path = Path(args.main_py).resolve()
    main_dir = main_py_path.parent

    if str(main_dir) not in sys.path:
        sys.path.insert(0, str(main_dir))

    original_cwd = os.getcwd()
    os.chdir(main_dir)

    try:
        spec = importlib.util.spec_from_file_location(
            "codegen_main", str(main_py_path)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print("\nRunning generated code with CPU golden comparison...\n")
        module.main()
        print("\nDone. Check output above for PCC comparison results.")
    except RuntimeError as e:
        if "pcc is" in str(e).lower():
            print(f"\nSTOPPED: CPU golden PCC failure detected:")
            print(f"  {e}")
        else:
            raise
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
