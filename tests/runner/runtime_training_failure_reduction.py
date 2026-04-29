# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate bounded runtime training-failure reduction bundles.

This tool creates review-ready reduction artifacts for training failures whose
current config metadata points to runtime or metal-style failures. It is a
bounded planning-driven implementation: it does not rerun tests, but it
normalizes the current runtime evidence and branches into draft packet or
attempt-log outputs using explicit heuristics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = (
    _PROJECT_ROOT
    / "tests"
    / "runner"
    / "test_config"
    / "torch"
    / "test_config_training_single_device.yaml"
)
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "artifacts" / "runtime_training_reduction"
_RUNTIME_KEYWORDS = (
    "FAILED_RUNTIME",
    "TT_FATAL",
    "L1",
    "DRAM",
    "Buffer must be allocated on device",
    "Test Hangs",
    "Out of Memory",
    "RuntimeError",
)
_DEBUG_LOG_SUFFIXES = (".log", ".txt")
_SAFE_PATH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass
class RuntimeReductionResult:
    test_id: str
    bringup_status: str | None
    reason: str | None
    classification: str
    owner_hint: str
    reduction_signal: str
    runtime_debug_captured: bool
    debug_log_path: str | None
    debug_evidence_path: str | None
    rerun_attempted: bool
    rerun_command: str | None
    rerun_log_path: str | None
    rerun_returncode: int | None
    force_run_skipped: bool
    next_manual_step: str
    output_dir: str
    draft_issue_path: str | None
    attempt_log_path: str | None


@dataclass
class RuntimeReductionSummary:
    config_path: str
    output_root: str
    selected_test_count: int
    tt_metal_draft_count: int
    tt_alchemist_draft_count: int
    attempt_log_count: int
    debug_evidence_count: int
    results: list[dict[str, Any]]


@dataclass
class DebugEvidence:
    source_path: str
    executing_operation_lines: list[str]
    runtime_signal_lines: list[str]
    ttnn_mlir_lines: list[str]


def detect_rerun_precondition_failure(log_path: Path | None) -> str | None:
    if log_path is None or not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8")
    if text.startswith("rerun precondition violation:"):
        return text.strip()
    if "TT-Metal installation could not be found" in text:
        return "rerun precondition violation: torch_xla could not locate TT-Metal installation"
    if (
        "Native library" in text
        and "pjrt_plugin_tt.so" in text
        and "does not exist" in text
    ):
        return "rerun precondition violation: torch_xla loaded source pjrt_plugin_tt without native plugin library"
    if "ImportError while loading conftest" in text and "ModuleNotFoundError" in text:
        missing = []
        for line in text.splitlines():
            if "ModuleNotFoundError" in line:
                missing.append(line.strip())
        detail = (
            "; ".join(missing) if missing else "pytest environment dependency missing"
        )
        return f"rerun precondition violation: {detail}"
    return None


def detect_rerun_invalid_execution(log_path: Path | None) -> str | None:
    if log_path is None or not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8")
    if "Defaulting to PJRT_DEVICE=CPU" in text:
        return "rerun executed with PJRT_DEVICE=CPU instead of TT hardware"
    if "SKIPPED" in text:
        return "rerun was skipped before TT runtime evidence was captured"
    if "TIMEOUT after" in text:
        return "rerun timed out before TT runtime evidence was captured"
    return None


def build_pytest_node_id(test_id: str) -> str:
    return f"tests/runner/test_models.py::test_all_models_torch[{test_id}]"


def build_rerun_command(pytest_bin: str, test_id: str) -> list[str]:
    return [pytest_bin, "-vv", "-s", build_pytest_node_id(test_id)]


def derive_python_bin(pytest_bin: str) -> str | None:
    pytest_path = Path(pytest_bin)
    if pytest_path.parent.name == "bin":
        candidate = pytest_path.parent / "python"
        if candidate.exists():
            return str(candidate)
    return None


def probe_rerun_environment(pytest_bin: str) -> str | None:
    python_bin = derive_python_bin(pytest_bin)
    if python_bin is None:
        return None

    probe = (
        "import importlib\n"
        "mods=['psutil','pytest','torch','torch_xla']\n"
        "missing=[]\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        '        missing.append(f"MISSING::{m}: {e}")\n'
        "print('\\n'.join(missing))\n"
    )
    completed = subprocess.run(
        [python_bin, "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        check=False,
    )
    output = completed.stdout.strip()
    missing_lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("MISSING::")
    ]
    if missing_lines:
        detail = "; ".join(line.replace("MISSING::", "", 1) for line in missing_lines)
        return f"rerun precondition violation: {detail}"
    return None


def load_training_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected mapping at {config_path}, got {type(data).__name__}"
        )
    test_config = data.get("test_config", data)
    if not isinstance(test_config, dict):
        raise ValueError(
            f"Expected 'test_config' mapping at {config_path}, got {type(test_config).__name__}"
        )
    return test_config


def sanitize_test_id(test_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", test_id)


def validate_safe_path_component(component: str, label: str) -> str:
    if component in {"", ".", ".."} or not _SAFE_PATH_COMPONENT_PATTERN.fullmatch(
        component
    ):
        raise ValueError(f"Unsafe {label} component: {component!r}")
    return component


def require_path_within(path: Path, root: Path, label: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must stay within {resolved_root}: {resolved_path}"
        ) from exc
    return resolved_path


def build_output_dir(output_root: Path, test_id: str) -> Path:
    safe_name = validate_safe_path_component(
        sanitize_test_id(test_id), "output directory"
    )
    resolved_root = output_root.expanduser().resolve()
    return require_path_within(
        resolved_root / safe_name, resolved_root, "output directory"
    )


def build_output_file(output_root: Path, file_name: str) -> Path:
    safe_name = validate_safe_path_component(file_name, "output file")
    resolved_root = output_root.expanduser().resolve()
    return require_path_within(resolved_root / safe_name, resolved_root, "output file")


def find_debug_log(debug_log_root: Path | None, test_id: str) -> Path | None:
    if debug_log_root is None or not debug_log_root.exists():
        return None

    if debug_log_root.is_file():
        return debug_log_root.expanduser().resolve()

    resolved_root = debug_log_root.expanduser().resolve()
    base_name = validate_safe_path_component(sanitize_test_id(test_id), "debug log")
    for suffix in _DEBUG_LOG_SUFFIXES:
        candidate = require_path_within(
            resolved_root / f"{base_name}{suffix}",
            resolved_root,
            "debug log",
        )
        if candidate.is_file():
            return candidate

    return None


def extract_debug_evidence(debug_log_path: Path) -> DebugEvidence | None:
    lines = debug_log_path.read_text(encoding="utf-8").splitlines()
    executing_operation_lines = [
        line.strip() for line in lines if "Executing operation" in line
    ]
    runtime_signal_lines = [
        line.strip()
        for line in lines
        if any(
            token in line
            for token in (
                "TT_FATAL",
                "beyond max L1 size",
                "Not enough space to allocate",
                "Buffer must be allocated on device",
                "RuntimeError",
                "Out of Memory",
            )
        )
    ]
    ttnn_mlir_lines = [
        line.rstrip()
        for line in lines
        if any(
            token in line for token in ("ttnn.", "tt.device", "ttnn::", "stablehlo.")
        )
    ]

    if (
        not executing_operation_lines
        and not runtime_signal_lines
        and not ttnn_mlir_lines
    ):
        return None

    return DebugEvidence(
        source_path=str(debug_log_path),
        executing_operation_lines=executing_operation_lines[:5],
        runtime_signal_lines=runtime_signal_lines[:8],
        ttnn_mlir_lines=ttnn_mlir_lines[:12],
    )


def run_bounded_rerun(
    *,
    pytest_bin: str,
    test_id: str,
    log_path: Path,
    timeout_sec: int,
    force_run_skipped: bool = False,
) -> tuple[int | None, str]:
    command = build_rerun_command(pytest_bin, test_id)
    env = dict(os.environ)
    env["TTMLIR_LOGGER_LEVEL"] = "DEBUG"
    env["TT_RUNTIME_DEBUG"] = "ON"
    if force_run_skipped:
        env["TT_XLA_FORCE_RUN_SKIPPED_TEST_IDS"] = test_id

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=timeout_sec,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        return completed.returncode, " ".join(command)
    except FileNotFoundError:
        log_path.write_text(
            f"pytest binary not found: {pytest_bin}\n", encoding="utf-8"
        )
        return None, " ".join(command)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log_path.write_text(
            output + f"\nTIMEOUT after {timeout_sec}s\n", encoding="utf-8"
        )
        return None, " ".join(command)


def classify_runtime_entry(
    bringup_status: str | None,
    reason: str | None,
    debug_evidence: DebugEvidence | None = None,
) -> tuple[str, str, str]:
    reason = reason or ""
    bringup_status = bringup_status or ""
    debug_signal_text = (
        " ".join(
            (
                debug_evidence.executing_operation_lines
                + debug_evidence.runtime_signal_lines
            )
        )
        if debug_evidence
        else ""
    )
    signal_text = f"{reason} {debug_signal_text}"

    if any(
        token in signal_text
        for token in (
            "TT_FATAL",
            "Buffer must be allocated on device",
            "Out of Memory",
            "beyond max L1 size",
            "Not enough space to allocate",
        )
    ):
        return (
            "draft_issue",
            "tt-metal",
            "runtime op/device or memory failure signature suggests tt-metal ownership",
        )
    if "FAILED_RUNTIME" in bringup_status and "Test Hangs" in reason:
        if debug_evidence and debug_evidence.executing_operation_lines:
            return (
                "draft_issue",
                "tt-alchemist",
                "runtime hang now has executing-operation evidence and is reduction-worthy for tt-alchemist follow-through",
            )
        return (
            "attempt_log",
            "unknown",
            "runtime hang requires real debug-log capture and cannot be confidently assigned from config-only evidence",
        )
    if "FAILED_RUNTIME" in bringup_status:
        if debug_evidence and (
            debug_evidence.executing_operation_lines
            or debug_evidence.runtime_signal_lines
            or debug_evidence.ttnn_mlir_lines
        ):
            return (
                "draft_issue",
                "tt-alchemist",
                "runtime failure has captured debug evidence and is reduction-worthy for tt-alchemist follow-through",
            )
        return (
            "draft_issue",
            "tt-alchemist",
            "runtime failure is reduction-worthy but needs op-level extraction and tt-alchemist follow-through",
        )
    return (
        "attempt_log",
        "unknown",
        "row does not match the bounded runtime-reduction selection strongly enough for a draft packet",
    )


def write_manifest(path: Path, result: RuntimeReductionResult) -> None:
    path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_summary(path: Path, summary: RuntimeReductionSummary) -> None:
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_review_report(path: Path, summary: RuntimeReductionSummary) -> None:
    tt_metal_rows = [
        result for result in summary.results if result["owner_hint"] == "tt-metal"
    ]
    tt_alchemist_rows = [
        result for result in summary.results if result["owner_hint"] == "tt-alchemist"
    ]
    attempt_rows = [result for result in summary.results if result["attempt_log_path"]]

    lines = [
        "# Runtime Training Failure Reduction Review Report",
        "",
        "## Summary",
        f"- selected tests: `{summary.selected_test_count}`",
        f"- tt-metal draft candidates: `{summary.tt_metal_draft_count}`",
        f"- tt-alchemist draft candidates: `{summary.tt_alchemist_draft_count}`",
        f"- attempt logs: `{summary.attempt_log_count}`",
        f"- debug evidence captured: `{summary.debug_evidence_count}`",
        "",
        "## TT-Metal Draft Candidates",
    ]
    if tt_metal_rows:
        for row in tt_metal_rows:
            lines.extend(
                [
                    f"- `{row['test_id']}`",
                    f"  - draft: `{row['draft_issue_path']}`",
                    f"  - reduction signal: `{row['reduction_signal']}`",
                    f"  - debug evidence: `{row['debug_evidence_path']}`",
                ]
            )
    else:
        lines.append("- None")

    lines.extend(["", "## TT-Alchemist Draft Candidates"])
    if tt_alchemist_rows:
        for row in tt_alchemist_rows:
            lines.extend(
                [
                    f"- `{row['test_id']}`",
                    f"  - draft: `{row['draft_issue_path']}`",
                    f"  - reduction signal: `{row['reduction_signal']}`",
                    f"  - debug evidence: `{row['debug_evidence_path']}`",
                ]
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Attempt Logs"])
    if attempt_rows:
        for row in attempt_rows:
            lines.extend(
                [
                    f"- `{row['test_id']}`",
                    f"  - attempt log: `{row['attempt_log_path']}`",
                    f"  - next step: `{row['next_manual_step']}`",
                    f"  - debug evidence: `{row['debug_evidence_path']}`",
                ]
            )
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_draft_issue(path: Path, result: RuntimeReductionResult) -> None:
    issue_title = "TT-Metal" if result.owner_hint == "tt-metal" else "TT-Alchemist"
    config_evidence = (
        result.reason
        or "No config reason recorded; draft is based on captured runtime debug evidence."
    )
    body = f"""# {issue_title} Runtime Reduction Draft

## Summary
`{result.test_id}` is a bounded runtime-reduction candidate with current config evidence:
`{config_evidence}`

## Runtime Classification
- owner hint: `{result.owner_hint}`
- classification: `{result.classification}`
- bringup status: `{result.bringup_status}`
- reduction signal: `{result.reduction_signal}`
- runtime debug captured: `{result.runtime_debug_captured}`

## Failure Context
- config source: `tests/runner/test_config/torch/test_config_training_single_device.yaml`
- current reason: `{config_evidence}`
- planned workflow target: metal/runtime reduction
- debug log: `{result.debug_log_path}`
- debug evidence: `{result.debug_evidence_path}`

## What Was Tried
- selected the row from the bounded runtime cohort based on runtime-oriented config evidence
- normalized the bringup status, reason, and any available runtime debug evidence into a runtime reduction signal
- mapped the row to a draft-owner path using explicit bounded heuristics

## Next Manual Step
- {result.next_manual_step}

## Review Gate
- draft only
- no external issue filing has occurred
"""
    path.write_text(body, encoding="utf-8")


def write_attempt_log(path: Path, result: RuntimeReductionResult) -> None:
    body = f"""workflow_path=runtime
test_id={result.test_id}
bringup_status={result.bringup_status}
reason={result.reason}
owner_hint={result.owner_hint}
reduction_signal={result.reduction_signal}
runtime_debug_captured={result.runtime_debug_captured}
debug_log_path={result.debug_log_path}
debug_evidence_path={result.debug_evidence_path}
rerun_attempted={result.rerun_attempted}
rerun_returncode={result.rerun_returncode}
force_run_skipped={result.force_run_skipped}

No draft issue packet was generated.

Attempted steps:
1. Read bounded runtime-style config row from the provided YAML.
2. Normalized the row into a runtime reduction signal.
3. Applied explicit owner heuristics for tt-metal, tt-alchemist, or no-draft fallback.

Blocked decision:
- {result.reduction_signal}

Next manual step:
- {result.next_manual_step}
"""
    path.write_text(body, encoding="utf-8")


def write_debug_evidence(path: Path, debug_evidence: DebugEvidence) -> None:
    lines = [
        "# Runtime Debug Evidence",
        "",
        f"- source log: `{debug_evidence.source_path}`",
        "",
        "## Executing Operation Lines",
    ]
    if debug_evidence.executing_operation_lines:
        lines.extend(
            [f"- `{line}`" for line in debug_evidence.executing_operation_lines]
        )
    else:
        lines.append("- None")

    lines.extend(["", "## Runtime Signal Lines"])
    if debug_evidence.runtime_signal_lines:
        lines.extend([f"- `{line}`" for line in debug_evidence.runtime_signal_lines])
    else:
        lines.append("- None")

    lines.extend(["", "## TTNN / MLIR Context"])
    if debug_evidence.ttnn_mlir_lines:
        lines.extend([f"- `{line}`" for line in debug_evidence.ttnn_mlir_lines])
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reduce_test_entry(
    *,
    test_id: str,
    entry: dict[str, Any],
    output_root: Path,
    debug_log_root: Path | None = None,
    execute_rerun: bool = False,
    pytest_bin: str = "pytest",
    rerun_timeout_sec: int = 900,
    force_run_skipped: bool = False,
) -> RuntimeReductionResult:
    bringup_status = entry.get("bringup_status")
    reason = entry.get("reason")
    output_dir = build_output_dir(output_root, test_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_log_path = find_debug_log(debug_log_root, test_id)
    rerun_attempted = False
    rerun_command = None
    rerun_log_path = None
    rerun_returncode = None

    if debug_log_path is None and execute_rerun:
        environment_precondition_failure = probe_rerun_environment(pytest_bin)
        if environment_precondition_failure is not None:
            rerun_attempted = False
            rerun_command = build_rerun_command(pytest_bin, test_id)
            rerun_log_file = output_dir / "runtime_rerun.log"
            rerun_log_file.write_text(
                environment_precondition_failure + "\n", encoding="utf-8"
            )
            rerun_log_path = str(rerun_log_file)
            debug_log_path = rerun_log_file
        else:
            rerun_attempted = True
            rerun_log_file = output_dir / "runtime_rerun.log"
            rerun_returncode, rerun_command = run_bounded_rerun(
                pytest_bin=pytest_bin,
                test_id=test_id,
                log_path=rerun_log_file,
                timeout_sec=rerun_timeout_sec,
                force_run_skipped=force_run_skipped,
            )
            rerun_log_path = str(rerun_log_file)
            debug_log_path = rerun_log_file

    debug_evidence = (
        extract_debug_evidence(debug_log_path)
        if debug_log_path and debug_log_path.exists()
        else None
    )
    rerun_precondition_failure = detect_rerun_precondition_failure(debug_log_path)
    rerun_invalid_execution = detect_rerun_invalid_execution(debug_log_path)
    classification, owner_hint, reduction_signal = classify_runtime_entry(
        bringup_status, reason, debug_evidence
    )

    if rerun_precondition_failure is not None:
        classification = "attempt_log"
        owner_hint = "unknown"
        reduction_signal = rerun_precondition_failure
    elif rerun_invalid_execution is not None:
        classification = "attempt_log"
        owner_hint = "unknown"
        reduction_signal = rerun_invalid_execution

    if classification == "draft_issue" and debug_evidence:
        next_manual_step = (
            "review the extracted runtime and TTNN/MLIR evidence, "
            "trim to the smallest repro-worthy snippet, then attach it before filing."
        )
    elif rerun_precondition_failure is not None:
        next_manual_step = (
            "install the missing pytest/runtime dependencies in the chosen TT-XLA environment, "
            "then rerun the bounded runtime capture."
        )
    elif (
        rerun_invalid_execution
        == "rerun executed with PJRT_DEVICE=CPU instead of TT hardware"
    ):
        next_manual_step = (
            "configure the rerun environment so the test runs on TT hardware and emits runtime debug markers, "
            "then rerun the bounded runtime capture."
        )
    elif (
        rerun_invalid_execution
        == "rerun timed out before TT runtime evidence was captured"
    ):
        next_manual_step = (
            "capture a staged or narrower debug run that emits TT runtime markers before the hang point, "
            "or choose a runtime row that reaches operation-level evidence within the bounded timeout."
        )
    elif rerun_invalid_execution is not None:
        next_manual_step = (
            "inspect the skip or early-runtime failure in the bounded rerun log, then decide whether to keep this row "
            "as an explicit blocker case or replace it with a row that reaches TT runtime markers."
        )
    elif execute_rerun and rerun_attempted:
        next_manual_step = (
            "inspect the bounded rerun log, confirm whether runtime debug markers were emitted, "
            "and refine the rerun command or environment before filing."
        )
    elif classification == "draft_issue":
        next_manual_step = (
            "re-run the failing test with TTMLIR_LOGGER_LEVEL=DEBUG and TT_RUNTIME_DEBUG=ON, "
            "capture the Executing operation line and enclosing TTNN MLIR, then attach the reduction evidence before filing."
        )
    else:
        next_manual_step = "capture real debug logs with TTMLIR_LOGGER_LEVEL=DEBUG and TT_RUNTIME_DEBUG=ON before assigning ownership."

    result = RuntimeReductionResult(
        test_id=test_id,
        bringup_status=bringup_status,
        reason=reason,
        classification=classification,
        owner_hint=owner_hint,
        reduction_signal=reduction_signal,
        runtime_debug_captured=debug_evidence is not None,
        debug_log_path=str(debug_log_path) if debug_log_path else None,
        debug_evidence_path=None,
        rerun_attempted=rerun_attempted,
        rerun_command=rerun_command,
        rerun_log_path=rerun_log_path,
        rerun_returncode=rerun_returncode,
        force_run_skipped=force_run_skipped,
        next_manual_step=next_manual_step,
        output_dir=str(output_dir),
        draft_issue_path=None,
        attempt_log_path=None,
    )

    if debug_evidence is not None:
        debug_evidence_path = output_dir / "debug_evidence.md"
        write_debug_evidence(debug_evidence_path, debug_evidence)
        result.debug_evidence_path = str(debug_evidence_path)

    if classification == "draft_issue":
        draft_issue_path = output_dir / "draft_issue.md"
        write_draft_issue(draft_issue_path, result)
        result.draft_issue_path = str(draft_issue_path)
    else:
        attempt_log_path = output_dir / "attempt.log"
        write_attempt_log(attempt_log_path, result)
        result.attempt_log_path = str(attempt_log_path)

    write_manifest(output_dir / "manifest.json", result)
    return result


def collect_selected_tests(
    config: dict[str, Any], requested_tests: list[str]
) -> list[str]:
    if requested_tests:
        missing = [test_id for test_id in requested_tests if test_id not in config]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise KeyError(f"Requested test ids not found in config: {missing_text}")
        return requested_tests

    selected = []
    for test_id, entry in config.items():
        if not isinstance(entry, dict):
            continue
        bringup_status = entry.get("bringup_status")
        reason = entry.get("reason", "")
        if bringup_status == "FAILED_RUNTIME" or any(
            token in reason for token in _RUNTIME_KEYWORDS
        ):
            selected.append(test_id)
    return selected


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--debug-log-root",
        type=Path,
        default=None,
        help="Optional directory of per-test runtime debug logs named after sanitized test ids, or a single debug log file.",
    )
    parser.add_argument(
        "--execute-rerun",
        action="store_true",
        help="If no matching debug log exists, attempt a bounded pytest rerun with TTMLIR_LOGGER_LEVEL=DEBUG and TT_RUNTIME_DEBUG=ON.",
    )
    parser.add_argument(
        "--pytest-bin",
        default="pytest",
        help="Pytest executable to use for --execute-rerun. Default: pytest",
    )
    parser.add_argument(
        "--rerun-timeout-sec",
        type=int,
        default=900,
        help="Timeout in seconds for each bounded rerun attempt.",
    )
    parser.add_argument(
        "--force-run-skipped",
        action="store_true",
        help=(
            "For --execute-rerun only: run selected NOT_SUPPORTED_SKIP rows by setting "
            "TT_XLA_FORCE_RUN_SKIPPED_TEST_IDS for the subprocess. This is intended for "
            "bounded debug capture and does not mutate source YAML."
        ),
    )
    parser.add_argument(
        "--test-id",
        action="append",
        default=[],
        help="Specific test id to reduce. Repeat for multiple tests. Defaults to all bounded runtime-style rows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    config = load_training_config(args.config)
    selected_tests = collect_selected_tests(config, args.test_id)
    results = [
        reduce_test_entry(
            test_id=test_id,
            entry=config[test_id],
            output_root=output_root,
            debug_log_root=args.debug_log_root,
            execute_rerun=args.execute_rerun,
            pytest_bin=args.pytest_bin,
            rerun_timeout_sec=args.rerun_timeout_sec,
            force_run_skipped=args.force_run_skipped,
        )
        for test_id in selected_tests
    ]

    summary = RuntimeReductionSummary(
        config_path=str(args.config),
        output_root=str(output_root),
        selected_test_count=len(results),
        tt_metal_draft_count=sum(
            1 for result in results if result.owner_hint == "tt-metal"
        ),
        tt_alchemist_draft_count=sum(
            1 for result in results if result.owner_hint == "tt-alchemist"
        ),
        attempt_log_count=sum(
            1 for result in results if result.attempt_log_path is not None
        ),
        debug_evidence_count=sum(
            1 for result in results if result.runtime_debug_captured
        ),
        results=[asdict(result) for result in results],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_summary(build_output_file(output_root, "summary.json"), summary)
    write_review_report(build_output_file(output_root, "review_report.md"), summary)

    print(
        "Generated "
        f"{len(selected_tests)} runtime reduction bundle(s) in {args.output_root} "
        f"(tt_metal={summary.tt_metal_draft_count}, tt_alchemist={summary.tt_alchemist_draft_count}, "
        f"attempt_logs={summary.attempt_log_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
