# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Measure device time of one decode step of a codegen_py ``main.py``.

Runs the existing ``<export-dir>/run -t`` script (which invokes ``python -m
tracy`` with the env it needs), then runs ``tt-perf-report`` scoped to the
``decode_1_start`` / ``decode_1_end`` signposts already wrapped in
``main.py``, and prints the total ``Device Time Sum`` (microseconds) for
the decode_1 step on stdout. Everything else goes to stderr.

This is the metric-producing half of the autoresearch Verify pipeline; the
PCC gate is ``verify_emitpy.py --pcc-threshold ...``.

Usage::

    python tests/benchmark/scripts/measure_emitpy.py <export-dir>
    # → 1234.567   (single number on stdout, microseconds)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SIGNPOST_START = "decode_1_start"
SIGNPOST_END = "decode_1_end"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess, streaming output to stderr; raise on non-zero exit."""
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=cwd, stdout=sys.stderr, stderr=sys.stderr, check=True)


def _latest_report_dir(export_dir: Path) -> Path:
    reports = export_dir / ".tracy_artifacts" / "reports"
    candidates = sorted(
        (p for p in reports.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No tracy report directories under {reports}. Did ./run -t fail?"
        )
    return candidates[0]


def _ops_perf_csv(report_dir: Path) -> Path:
    matches = list(report_dir.glob("ops_perf_results_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No ops_perf_results_*.csv in {report_dir}. tracy did not produce "
            f"a CSV — check the report dir for partial outputs."
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run tracy + tt-perf-report on a codegen_py main.py and "
        "print total Device Time Sum for decode_1.",
    )
    parser.add_argument(
        "export_dir",
        type=Path,
        help="Directory containing main.py, run, tensors/, .tracy_artifacts/ "
        "(the --codegen-py-export-path passed to the benchmark).",
    )
    parser.add_argument(
        "--start-signpost",
        default=SIGNPOST_START,
        help=f"Start signpost (default: {SIGNPOST_START}).",
    )
    parser.add_argument(
        "--end-signpost",
        default=SIGNPOST_END,
        help=f"End signpost (default: {SIGNPOST_END}).",
    )
    args = parser.parse_args()

    export_dir: Path = args.export_dir.resolve()
    if not (export_dir / "main.py").is_file():
        print(f"error: {export_dir}/main.py not found.", file=sys.stderr)
        return 2
    if not (export_dir / "run").is_file():
        print(f"error: {export_dir}/run not found.", file=sys.stderr)
        return 2

    _run(["./run", "-t"], cwd=export_dir)

    report_dir = _latest_report_dir(export_dir)
    print(f"# report_dir: {report_dir}", file=sys.stderr)

    csv = _ops_perf_csv(report_dir)
    summary_prefix = report_dir / "summary"

    _run(
        [
            "tt-perf-report",
            str(csv),
            "--start-signpost",
            args.start_signpost,
            "--end-signpost",
            args.end_signpost,
            "--summary-file",
            str(summary_prefix),
        ]
    )

    summary_csv = report_dir / "summary.csv"
    if not summary_csv.is_file():
        print(
            f"error: tt-perf-report did not produce {summary_csv}.",
            file=sys.stderr,
        )
        return 2

    df = pd.read_csv(summary_csv)
    if "Device Time Sum" not in df.columns:
        print(
            f"error: 'Device Time Sum' column missing from {summary_csv}; "
            f"available: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    total = float(df["Device Time Sum"].sum())
    print(f"{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
