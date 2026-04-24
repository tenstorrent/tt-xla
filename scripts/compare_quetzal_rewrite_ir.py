#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare TT-XLA IR with and without quetzal-inspired FX rewrites.

This script runs small Torch graphs through the TT-XLA backend twice:

  1. quetzal rewrites explicitly disabled
  2. quetzal rewrites enabled

It exports MLIR for both runs, counts high-signal operation tokens in the IR,
and writes a JSON summary next to the exported artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_DIR = REPO_ROOT / "python_package"

if PYTHON_PACKAGE_DIR.exists():
    sys.path.insert(0, str(PYTHON_PACKAGE_DIR))


CASE_ALL = "all"
CASE_DECOMPOSED_TANH_GELU = "decomposed_tanh_gelu"
CASE_MANUAL_SDPA = "manual_sdpa"

MODE_OFF = "off"
MODE_ON = "on"

IR_TOKENS = (
    "tenstorrent.gelu_tanh",
    "tenstorrent.gelu",
    "ttir.gelu",
    "ttnn.gelu",
    "stablehlo.tanh",
    "stablehlo.power",
    "stablehlo.multiply",
    "tenstorrent.scaled_dot_product_attention",
    "ttir.scaled_dot_product_attention",
    "ttnn.scaled_dot_product_attention",
    "stablehlo.dot_general",
    "stablehlo.exponential",
    "ttir.matmul",
    "ttnn.matmul",
    "ttir.softmax",
    "ttnn.softmax",
)

STAGES = (
    "shlo_set_mesh_attr",
    "shlo_frontend",
    "shlo_compiler",
    "ttnn_runtime",
    "vhlo",
    "shlo",
    "ttir",
    "ttnn",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    dtype_name: str
    shapes: tuple[tuple[int, ...], ...]
    build_model: Callable[[Any], Any]
    expected_positive_tokens: tuple[str, ...]


@dataclass
class IRSummary:
    export_path: Path
    files: list[Path]
    counts: Counter[str]
    counts_by_stage: dict[str, Counter[str]]

    def to_json(self) -> dict[str, Any]:
        return {
            "export_path": str(self.export_path),
            "files": [str(path) for path in self.files],
            "counts": dict(sorted(self.counts.items())),
            "counts_by_stage": {
                stage: dict(sorted(counts.items()))
                for stage, counts in sorted(self.counts_by_stage.items())
            },
        }


@dataclass
class RunResult:
    case_name: str
    mode: str
    rewrite_passes: str
    export_path: Path
    model_name: str
    summary: IRSummary
    runtime_error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "case": self.case_name,
            "mode": self.mode,
            "rewrite_passes": self.rewrite_passes,
            "export_path": str(self.export_path),
            "model_name": self.model_name,
            "runtime_error": self.runtime_error,
            "summary": self.summary.to_json(),
        }


def build_decomposed_tanh_gelu_model(torch_module):
    class DecomposedTanhGELU(torch_module.nn.Module):
        def forward(self, x):
            return 0.5 * x * (
                1.0
                + torch_module.tanh(
                    0.7978845608028654 * (x + 0.044715 * x.pow(3.0))
                )
            )

    return DecomposedTanhGELU()


def build_manual_sdpa_model(torch_module):
    class ManualSDPA(torch_module.nn.Module):
        def forward(self, query, key, value):
            scale = query.shape[-1] ** -0.5
            scores = torch_module.matmul(query, key.transpose(-2, -1)).mul(scale)
            weights = torch_module.softmax(scores, dim=-1)
            return torch_module.matmul(weights, value)

    return ManualSDPA()


CASE_SPECS = {
    CASE_DECOMPOSED_TANH_GELU: CaseSpec(
        name=CASE_DECOMPOSED_TANH_GELU,
        dtype_name="float32",
        shapes=((64, 256),),
        build_model=build_decomposed_tanh_gelu_model,
        expected_positive_tokens=(
            "tenstorrent.gelu_tanh",
            "ttir.gelu",
            "ttnn.gelu",
        ),
    ),
    CASE_MANUAL_SDPA: CaseSpec(
        name=CASE_MANUAL_SDPA,
        dtype_name="bfloat16",
        shapes=((1, 8, 32, 64), (1, 8, 32, 64), (1, 8, 32, 64)),
        build_model=build_manual_sdpa_model,
        expected_positive_tokens=(
            "tenstorrent.scaled_dot_product_attention",
            "ttir.scaled_dot_product_attention",
            "ttnn.scaled_dot_product_attention",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export and compare TT-XLA MLIR with quetzal-inspired rewrites "
            "disabled and enabled."
        )
    )
    parser.add_argument(
        "--case",
        choices=(CASE_ALL, *CASE_SPECS.keys()),
        default=CASE_ALL,
        help="Graph case to run.",
    )
    parser.add_argument(
        "--rewrite-passes",
        default="all",
        help=(
            "Value passed as tt_quetzal_rewrite_passes for the enabled run. "
            "Use 'all' or a comma-separated list such as 'fuse_gelu,reconstruct_sdpa'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "quetzal_ir_compare",
        help="Directory where a timestamped comparison subdirectory is created.",
    )
    parser.add_argument(
        "--system-desc",
        type=Path,
        default=None,
        help=(
            "Optional .ttsys descriptor for TT compile-only mode. This must be "
            "provided to compile without live hardware."
        ),
    )
    parser.add_argument(
        "--device-type",
        default="TT",
        help="torch_xla runtime device type.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero if the enabled run does not add expected fused IR tokens.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next case if a compile fails.",
    )
    return parser.parse_args()


def selected_cases(case_name: str) -> list[CaseSpec]:
    if case_name == CASE_ALL:
        return [CASE_SPECS[name] for name in sorted(CASE_SPECS)]
    return [CASE_SPECS[case_name]]


def stage_from_path(path: Path) -> str:
    for stage in STAGES:
        if path.name.startswith(f"{stage}_"):
            return stage
    return "unknown"


def summarize_export(export_path: Path) -> IRSummary:
    files = sorted((export_path / "irs").glob("*.mlir"))
    counts: Counter[str] = Counter()
    counts_by_stage: dict[str, Counter[str]] = {}

    for path in files:
        stage = stage_from_path(path)
        stage_counts = counts_by_stage.setdefault(stage, Counter())
        text = path.read_text(encoding="utf-8", errors="replace")

        for token in IR_TOKENS:
            count = text.count(token)
            if count:
                counts[token] += count
                stage_counts[token] += count

    return IRSummary(
        export_path=export_path,
        files=files,
        counts=counts,
        counts_by_stage=counts_by_stage,
    )


def dtype_from_name(torch_module, dtype_name: str):
    try:
        return getattr(torch_module, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype name: {dtype_name}") from exc


def ensure_runtime(system_desc: Path | None):
    try:
        import torch
        import torch_xla
        import torch_xla.runtime as xr
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import torch_xla. Install the TT-XLA Python requirements "
            "in a compatible environment before running IR comparison."
        ) from exc

    if system_desc is not None:
        from ttxla_tools import enable_compile_only

        enable_compile_only(str(system_desc))

    # Import tt_torch only after optional compile-only setup so backend
    # registration does not accidentally initialize the PJRT client first.
    import tt_torch  # noqa: F401

    return torch, torch_xla, xr


def run_one(
    case: CaseSpec,
    mode: str,
    rewrite_passes: str,
    run_root: Path,
    torch_module,
    torch_xla_module,
    device,
    compile_only: bool,
) -> RunResult:
    torch_module._dynamo.reset()
    torch_module.manual_seed(0)

    export_path = run_root / case.name / mode
    export_path.mkdir(parents=True, exist_ok=True)

    model_name = f"{case.name}_{mode}"
    torch_xla_module.set_custom_compile_options(
        {
            "export_path": str(export_path),
            "export_model_name": model_name,
        }
    )

    dtype = dtype_from_name(torch_module, case.dtype_name)
    model = case.build_model(torch_module).to(dtype=dtype).to(device)
    inputs = [
        torch_module.randn(*shape, dtype=dtype, device=device) for shape in case.shapes
    ]

    backend_options = {
        "tt_quetzal_rewrite_passes": "none" if mode == MODE_OFF else rewrite_passes
    }

    runtime_error = None
    try:
        compiled = torch_module.compile(model, backend="tt", options=backend_options)
        output = compiled(*inputs)
        if isinstance(output, torch_module.Tensor):
            outputs = (output,)
        elif isinstance(output, (tuple, list)):
            outputs = tuple(item for item in output if isinstance(item, torch_module.Tensor))
        else:
            outputs = ()

        if outputs:
            torch_xla_module.sync()
        else:
            torch_xla_module.sync()
    except RuntimeError as exc:
        summary = summarize_export(export_path)
        if not compile_only or not summary.files:
            raise
        runtime_error = str(exc)

    summary = summarize_export(export_path)
    return RunResult(
        case_name=case.name,
        mode=mode,
        rewrite_passes=backend_options["tt_quetzal_rewrite_passes"],
        export_path=export_path,
        model_name=model_name,
        summary=summary,
        runtime_error=runtime_error,
    )


def token_sum(summary: IRSummary, tokens: tuple[str, ...]) -> int:
    return sum(summary.counts[token] for token in tokens)


def expected_signal_present(
    case: CaseSpec, off_summary: IRSummary, on_summary: IRSummary
) -> bool:
    return token_sum(on_summary, case.expected_positive_tokens) > token_sum(
        off_summary, case.expected_positive_tokens
    )


def all_counted_tokens(results: list[RunResult]) -> list[str]:
    tokens = set()
    for result in results:
        tokens.update(result.summary.counts.keys())
    return [token for token in IR_TOKENS if token in tokens]


def print_case_report(case: CaseSpec, off: RunResult, on: RunResult) -> None:
    print(f"\nCase: {case.name}")
    print(f"  off export: {off.export_path} ({len(off.summary.files)} MLIR files)")
    print(f"  on  export: {on.export_path} ({len(on.summary.files)} MLIR files)")

    if off.runtime_error:
        print("  off runtime: compile-only artifacts collected after execution error")
    if on.runtime_error:
        print("  on  runtime: compile-only artifacts collected after execution error")

    tokens = all_counted_tokens([off, on])
    if not tokens:
        print("  No tracked IR tokens found.")
        return

    print("  Token counts:")
    print("    token                                         off     on  delta")
    for token in tokens:
        off_count = off.summary.counts[token]
        on_count = on.summary.counts[token]
        print(f"    {token:<42} {off_count:>5} {on_count:>6} {on_count - off_count:>6}")

    signal = "yes" if expected_signal_present(case, off.summary, on.summary) else "no"
    print(f"  Expected fused-token increase: {signal}")


def write_summary(run_root: Path, results: list[RunResult]) -> Path:
    summary_path = run_root / "summary.json"
    payload = {
        "run_root": str(run_root),
        "results": [result.to_json() for result in results],
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    run_root = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_root.mkdir(parents=True, exist_ok=False)

    if args.system_desc is not None:
        args.system_desc = args.system_desc.resolve()
        if not args.system_desc.exists():
            print(f"System descriptor does not exist: {args.system_desc}", file=sys.stderr)
            return 2

    try:
        torch_module, torch_xla_module, xr = ensure_runtime(args.system_desc)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    os.environ.setdefault("XLA_HLO_DEBUG", "1")
    xr.set_device_type(args.device_type)
    device = torch_xla_module.device()

    all_results: list[RunResult] = []
    failed_cases: list[str] = []

    for case in selected_cases(args.case):
        try:
            off = run_one(
                case,
                MODE_OFF,
                args.rewrite_passes,
                run_root,
                torch_module,
                torch_xla_module,
                device,
                compile_only=args.system_desc is not None,
            )
            on = run_one(
                case,
                MODE_ON,
                args.rewrite_passes,
                run_root,
                torch_module,
                torch_xla_module,
                device,
                compile_only=args.system_desc is not None,
            )
        except Exception as exc:
            failed_cases.append(case.name)
            print(f"\nCase failed: {case.name}", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            if not args.keep_going:
                summary_path = write_summary(run_root, all_results)
                print(f"\nSummary JSON: {summary_path}")
                return 1
            continue

        all_results.extend([off, on])
        print_case_report(case, off, on)

        if args.strict and not expected_signal_present(case, off.summary, on.summary):
            failed_cases.append(case.name)

    summary_path = write_summary(run_root, all_results)
    print(f"\nSummary JSON: {summary_path}")

    if failed_cases:
        print(f"Failed cases: {', '.join(failed_cases)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
