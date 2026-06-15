#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

ISSUE_NUMBER = 5009
ISSUE_URL = "https://github.com/tenstorrent/tt-xla/issues/5009"
PRD_ID = "PRD-009"
DEFAULT_OUTPUT_ROOT = Path("artifacts/prd-009/ttxla-profile")
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_MAX_OUTPUT_TOKENS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_LAYERS = 1
DEFAULT_INPUT_SEQUENCE_LENGTH = 0
DEFAULT_MAX_RAW_ARTIFACT_BYTES = 100_000_000
DEFAULT_IRD_RUN_BUDGET_BUFFER_SECONDS = 300
DEFAULT_COLLECTOR_JOB_ID = "0"
PERF_REPORT_CSV_PATTERNS = (
    "ops_perf_results_*.csv",
    "ops_perf_results.csv",
    "cpp_device_perf_report.csv",
)
TT_PERF_REPORT_CSV_PATTERNS = (
    "ops_perf_results_*.csv",
    "ops_perf_results.csv",
)
TT_PERF_REPORT_REQUIRED_COLUMNS = ("OP TYPE",)
DEFAULT_BENCHMARK_FILES = [
    Path("tests/benchmark/test_llms.py"),
    Path("tests/benchmark/test_encoders.py"),
    Path("tests/benchmark/test_vision.py"),
    Path("tests/benchmark/resnet_jax_benchmark.py"),
]
RUNNER_TORCH_INFERENCE_SOURCE = "tests/runner/test_models.py"
RUNNER_TORCH_INFERENCE_TEST = "test_all_models_torch"
RUNNER_TORCH_INFERENCE_FAMILY = "runner_torch_inference"

REQS = [
    (
        "REQ-F-001",
        "Discover or ingest the complete model list from the TTXLA benchmarking test directory.",
    ),
    (
        "REQ-F-002",
        "Each model must receive a deterministic run identity, model identity, hardware target, and benchmark source path.",
    ),
    (
        "REQ-F-003",
        "The pipeline must execute the profiling command or record an explicit skip/blocker reason.",
    ),
    (
        "REQ-F-004",
        "The pipeline must collect IR artifacts for every model where IR generation succeeds.",
    ),
    (
        "REQ-F-005",
        "The pipeline must run tt-perf-report or record why tt-perf-report could not run.",
    ),
    (
        "REQ-F-006",
        "Each model must emit status.json with terminal state, taxonomy classification, commands, artifact paths, and next action.",
    ),
    (
        "REQ-F-007",
        "The dashboard must rank slowest operations globally, by model, and by op type, with links to IR and perf artifacts.",
    ),
    (
        "REQ-F-008",
        "The final stakeholder report must be generated from a Claude refinement packet and emitted as HTML per PDR-001.",
    ),
    (
        "REQ-F-009",
        "The dashboard must support searching or filtering by model identity, op type, profile status, and taxonomy classification.",
    ),
    (
        "REQ-F-010",
        "The pipeline must emit Superset collector-compatible benchmark report artifacts for slow-op rows.",
    ),
]

ENVIRONMENT_FAILURE_HINTS = (
    "module not found",
    "no module named",
    "command not found",
    "permission denied",
    "file not found",
    "no such file or directory",
    "filesystem error",
    "cannot create directories",
    "missing dependency",
    "silicon_sysmem_manager",
    "pin_or_map_sysmem_to_device",
)

MODEL_FAILURE_HINTS = (
    "assertionerror",
    "pcc comparison failed",
    "comparison failed",
    "runtimeerror",
    "tt_fatal",
    "fatal error",
    "ttmlir compilation",
    "frontend conversion",
    "unsupported operator",
    "device crash",
    "bad statusor access",
    "traceback",
)

PIPELINE_ERROR_HINTS = (
    "pytest: error:",
    "unrecognized arguments:",
    "usage: pytest",
)

SKIP_HINTS = (
    "skipped",
    "xfailed",
    "xfail",
)

PYTEST_PASS_PATTERN = re.compile(
    r"(?m)(^PASSED\s*$|=+\s+\d+\s+passed\b|::[^\n]+\s+PASSED(?:\s|$))",
    re.IGNORECASE,
)

TAXONOMY_VALIDATED_PASS = "validated_pass"
TAXONOMY_VALIDATED_FAIL = "validated_fail"
TAXONOMY_MODEL_FAILURE = "model_failure"
TAXONOMY_ENVIRONMENT_FAILURE = "environment_failure"
TAXONOMY_PIPELINE_ERROR = "pipeline_error"
TAXONOMY_NOT_RUN = "not_run"
TAXONOMY_NOT_STARTED = "not_started"
TAXONOMY_SKIPPED_WITH_REASON = "skipped_with_reason"
TAXONOMY_COMPILED_ONLY = "compiled_only"

TERMINAL_STATE_PASSED = "passed"
TERMINAL_STATE_FAILED = "failed"
TERMINAL_STATE_BLOCKED = "blocked"
TERMINAL_STATE_SKIPPED = "skipped"
TERMINAL_STATE_PARTIAL = "partial"

RUN_STATUS_NOT_RUN = "not_run"
RUN_STATUS_BLOCKED = "blocked"
RUN_STATUS_UNKNOWN = "unknown"

CSV_DURATION_ALIASES = (
    "DEVICE FW DURATION [ns]",
    "DEVICE KERNEL DURATION [ns]",
    "HOST DURATION [ns]",
    "duration_us",
    "duration_ms",
    "duration_ns",
    "elapsed_us",
    "elapsed_ms",
    "elapsed_ns",
    "time_us",
    "time_ms",
    "time_ns",
    "latency_us",
    "latency_ms",
    "latency_ns",
    "duration",
    "elapsed",
    "time",
)

CSV_OP_ALIASES = ("OP CODE", "op_name", "operation", "op", "name", "kernel")
CSV_MODEL_ALIASES = ("model", "model_name", "model_id", "model_identity")
CSV_OP_TYPE_ALIASES = ("OP TYPE", "op_type", "type", "category", "kind")


@dataclass
class DiscoveryEntry:
    run_identity: str
    nodeid: str
    source_path: str
    test_name: str
    benchmark_family: str
    model_identity: str
    artifact_slug: str
    source_branch: str = ""


@dataclass
class CommandResult:
    stage: str
    command: list[str]
    cwd: str
    returncode: Optional[int]
    timed_out: bool
    start_time: str
    end_time: str
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    note: str = ""

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0


@dataclass
class IrdReservation:
    reservation_id: str
    target_host: str
    raw_stdout: str
    raw_stderr: str


@dataclass
class ProfilePaths:
    profile_dir: Path
    perf_dir: Path
    ir_dir: Path
    trace_dir: Path
    benchmark_output: Path
    run_log: Path
    compile_log: Path
    perf_output: Path
    perf_input: Path
    slow_ops_path: Path


@dataclass
class PerfReportOutcome:
    result: Optional[CommandResult]
    ok: bool
    reason: str
    command: list[str]
    csv_source: Optional[Path]
    csv_recorded: str


@dataclass
class RequirementCoverageContext:
    run_dir: Path
    model_manifest_path: Path
    model_manifest: dict[str, Any]
    model_count: int
    status_count: int
    status_paths: list[Path]
    status_files_exist: bool
    perf_report_count: int
    ir_count: int
    dashboard_path: Path
    packet_path: Path
    report_path: Path
    dashboard_text: str
    superset_export_dir: Path
    superset_export_paths: list[Path]
    statuses: list[dict[str, Any]]


@dataclass
class LocalPipelineContext:
    root: Path
    run_dir: Path
    ir_dump_root: Path
    python_bin: str
    pytest_command: str
    tracy_bin: str
    tt_perf_report_bin: str
    command_trace_path: Path
    run_deadline: Optional[float]
    environment: dict[str, Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime(
        "run-5009-%Y%m%d-%H%M%S-"
    ) + secrets.token_hex(2)


def slugify(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace(os.sep, "_")
    text = text.replace("::", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "na"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def render_template(value: str, variables: dict[str, str]) -> str:
    rendered = value
    for key, replacement in variables.items():
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


def split_template_command(template: str, variables: dict[str, str]) -> list[str]:
    return shlex.split(render_template(template, variables))


def benchmark_family_from_source(source_path: str) -> str:
    name = Path(source_path).name
    if name == "test_llms.py":
        return "llm"
    if name == "test_encoders.py":
        return "encoder"
    if name == "test_vision.py":
        return "vision"
    if name == "resnet_jax_benchmark.py":
        return "jax"
    return Path(source_path).stem


def benchmark_files(repo: Path) -> list[Path]:
    return [repo / relative for relative in DEFAULT_BENCHMARK_FILES]


def selected_benchmark_files(repo: Path, values: Iterable[str]) -> list[Path]:
    selected = [value for value in (item.strip() for item in values) if value]
    if not selected:
        return benchmark_files(repo)
    return [
        path if path.is_absolute() else repo / path
        for path in (Path(value) for value in selected)
    ]


def repo_subprocess_environment(
    repo: Path, base_env: Optional[dict[str, str]] = None
) -> dict[str, str]:
    env = dict(base_env or os.environ.copy())
    repo_paths = [str(repo / "tests"), str(repo)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        repo_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(repo_paths)
    return env


def collect_command(python_bin: str, files: Iterable[Path]) -> list[str]:
    return [
        python_bin,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        *[str(path) for path in files],
    ]


def profile_command(
    tracy_bin: str,
    pytest_command: str,
    nodeid: str,
    profile_dir: Path,
    benchmark_output: Path,
    ir_dump_root: Path,
    benchmark_args: list[str],
) -> list[str]:
    base_command = [
        *split_command_expr(tracy_bin),
        "-p",
        "-r",
        "--sync-host-device",
        "-o",
        str(profile_dir / "tracy"),
        "-m",
        pytest_command,
        "-svv",
        nodeid,
        "--dump-irs",
        "--dump-irs-dir",
        str(ir_dump_root),
    ]
    if nodeid.startswith(f"{RUNNER_TORCH_INFERENCE_SOURCE}::"):
        return [
            *base_command,
            "--perf-report-dir",
            str(profile_dir / "perf-report"),
            "--perf-id",
            slugify(nodeid),
            *benchmark_args,
        ]
    return [
        *base_command,
        "--output-file",
        str(benchmark_output),
        *benchmark_args,
    ]


def benchmark_args_for_entry(
    entry: DiscoveryEntry,
    benchmark_kwargs: dict[str, int],
) -> list[str]:
    family = entry.benchmark_family
    args: list[str] = []
    if family == "llm":
        args.extend(["--batch-size", str(benchmark_kwargs["batch_size"])])
        args.extend(["--num-layers", str(benchmark_kwargs["num_layers"])])
        args.extend(["--max-output-tokens", str(benchmark_kwargs["max_output_tokens"])])
    elif family == "encoder":
        args.extend(["--batch-size", str(benchmark_kwargs["batch_size"])])
        args.extend(["--num-layers", str(benchmark_kwargs["num_layers"])])
        if benchmark_kwargs["input_sequence_length"] > 0:
            args.extend(
                [
                    "--input-sequence-length",
                    str(benchmark_kwargs["input_sequence_length"]),
                ]
            )
    elif family == "jax":
        args.extend(["--batch-size", str(benchmark_kwargs["batch_size"])])
    return args


def perf_report_command(tt_perf_report_bin: str, csv_path: Path) -> list[str]:
    return [*split_command_expr(tt_perf_report_bin), str(csv_path)]


def split_command_expr(command_expr: str) -> list[str]:
    return shlex.split(command_expr) if command_expr else []


def command_expr_available(command_expr: str) -> bool:
    parts = split_command_expr(command_expr)
    if not parts:
        return False
    executable = parts[0]
    if os.sep in executable:
        return Path(executable).exists()
    return shutil.which(executable) is not None


def run_subprocess(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    stage: str,
    command_trace_path: Path,
    timeout_seconds: int,
    env: Optional[dict[str, str]] = None,
) -> CommandResult:
    ensure_dir(stdout_path.parent)
    ensure_dir(stderr_path.parent)

    started_at = time.monotonic()
    start_wall = now_iso()
    timed_out = False
    returncode: Optional[int] = None
    note = ""

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env or os.environ.copy(),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        except OSError as exc:
            note = f"failed to start command: {command[0]} ({exc})"
            stdout_handle.write("")
            stderr_handle.write(note + "\n")
            end_wall = now_iso()
            duration = time.monotonic() - started_at
            record = CommandResult(
                stage=stage,
                command=command,
                cwd=str(cwd),
                returncode=127,
                timed_out=False,
                start_time=start_wall,
                end_time=end_wall,
                duration_seconds=duration,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                note=note,
            )
            append_command_trace(command_trace_path, record)
            return record
        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            note = f"timed out after {timeout_seconds} seconds"
            proc.kill()
            returncode = proc.wait()

    end_wall = now_iso()
    duration = time.monotonic() - started_at
    record = CommandResult(
        stage=stage,
        command=command,
        cwd=str(cwd),
        returncode=returncode,
        timed_out=timed_out,
        start_time=start_wall,
        end_time=end_wall,
        duration_seconds=duration,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        note=note,
    )
    append_command_trace(command_trace_path, record)
    return record


def append_command_trace(path: Path, record: CommandResult) -> None:
    ensure_dir(path.parent)
    payload = asdict(record)
    payload["command"] = shell_join(record.command)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError as exc:
        print(
            f"warning: failed to append command trace {path}: {exc}",
            file=sys.stderr,
        )


def prune_large_raw_artifacts(
    profile_dir: Path, max_bytes: int
) -> list[dict[str, Any]]:
    if max_bytes <= 0:
        return []
    candidates = [
        profile_dir / "tracy" / ".logs" / "tracy_ops_times.csv",
        profile_dir / "tracy" / ".logs" / "profile_log_device.csv",
        profile_dir / "tracy" / ".logs" / "tracy_profile_log_host.tracy",
    ]
    candidates.extend(
        sorted((profile_dir / "tracy" / "reports").glob("*/profile_log_device.csv"))
    )
    candidates.extend(
        sorted(
            (profile_dir / "tracy" / "reports").glob("*/tracy_profile_log_host.tracy")
        )
    )
    pruned = []
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        size = path.stat().st_size
        if size <= max_bytes:
            continue
        path.unlink()
        pruned.append({"path": str(path), "bytes": size})
    return pruned


def parse_collect_output(output: str, run_id: str) -> list[DiscoveryEntry]:
    entries: list[DiscoveryEntry] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith(("tests/benchmark/", RUNNER_TORCH_INFERENCE_SOURCE)):
            continue
        token = line.split()[0]
        if "::" not in token:
            continue
        if token in seen:
            continue
        seen.add(token)
        source_path, test_name = token.split("::", 1)
        model_identity = test_name.split("[", 1)[0]
        artifact_slug = slugify(token)
        entries.append(
            DiscoveryEntry(
                run_identity=f"{run_id}-{len(entries)+1:04d}",
                nodeid=token,
                source_path=source_path,
                test_name=test_name,
                benchmark_family=benchmark_family_from_source(source_path),
                model_identity=model_identity,
                artifact_slug=artifact_slug,
            )
        )
    return entries


def runner_torch_nodeid(test_case_id: str) -> str:
    return (
        f"{RUNNER_TORCH_INFERENCE_SOURCE}::{RUNNER_TORCH_INFERENCE_TEST}"
        f"[{test_case_id}-single_device-inference]"
    )


def discovery_entry_from_nvidia_row(
    row: dict[str, Any],
    run_id: str,
    index: int,
) -> Optional[DiscoveryEntry]:
    test_case_id = str(row.get("test_case_id") or "").strip()
    if not test_case_id:
        return None
    nodeid = runner_torch_nodeid(test_case_id)
    display_name = (
        row.get("display_name")
        or row.get("model_id")
        or row.get("pretrained_model_name")
        or test_case_id
    )
    return DiscoveryEntry(
        run_identity=f"{run_id}-{index:04d}",
        nodeid=nodeid,
        source_path=RUNNER_TORCH_INFERENCE_SOURCE,
        test_name=f"{RUNNER_TORCH_INFERENCE_TEST}[{test_case_id}-single_device-inference]",
        benchmark_family=RUNNER_TORCH_INFERENCE_FAMILY,
        model_identity=str(display_name),
        artifact_slug=slugify(nodeid),
        source_branch=str(row.get("source_branch") or "").strip(),
    )


def load_nvidia_cohort_entries(path: Path, run_id: str) -> list[DiscoveryEntry]:
    payload = load_json(path)
    entries: list[DiscoveryEntry] = []
    seen_test_case_ids: set[str] = set()
    for row in payload.get("models", []) or []:
        if not isinstance(row, dict):
            continue
        test_case_id = str(row.get("test_case_id") or "").strip()
        if not test_case_id or test_case_id in seen_test_case_ids:
            continue
        entry = discovery_entry_from_nvidia_row(row, run_id, len(entries) + 1)
        if entry is None:
            continue
        seen_test_case_ids.add(test_case_id)
        entries.append(entry)
    return entries


def nvidia_cohort_discovery_result(
    repo: Path, cohort_path: Path, entries: list[DiscoveryEntry]
) -> CommandResult:
    created_at = now_iso()
    note = f"loaded {len(entries)} NVIDIA/SILICON_PASS cohort rows by test_case_id"
    return CommandResult(
        stage="discover",
        command=["load-nvidia-cohort-json", str(cohort_path)],
        cwd=str(repo),
        returncode=0,
        timed_out=False,
        start_time=created_at,
        end_time=created_at,
        duration_seconds=0.0,
        stdout_path="",
        stderr_path="",
        note=note,
    )


def collect_runner_models_for_nvidia_cohort(
    repo: Path,
    run_id: str,
    python_bin: str,
    command_trace_path: Path,
    timeout_seconds: int = 900,
) -> tuple[list[DiscoveryEntry], CommandResult]:
    command = collect_command(python_bin, [Path(RUNNER_TORCH_INFERENCE_SOURCE)])
    tmp_dir = ensure_dir(repo / ".tmp" / "ttxla-profile-discovery")
    stdout_path = tmp_dir / f"collect-nvidia-runner-{run_id}.out"
    stderr_path = tmp_dir / f"collect-nvidia-runner-{run_id}.err"
    result = run_subprocess(
        command=command,
        cwd=repo,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stage="discover-nvidia-runner",
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
        env=repo_subprocess_environment(repo),
    )
    collected = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    entries = parse_collect_output(collected, run_id)
    result.note = (
        f"collected {len(entries)} TT runner nodes before NVIDIA cohort filtering"
    )
    return entries, result


def source_branch_for_entries(entries: list[DiscoveryEntry]) -> str:
    branches = {entry.source_branch for entry in entries if entry.source_branch}
    if len(branches) == 1:
        return next(iter(branches))
    return ""


def checkout_nvidia_source_branch(
    repo: Path,
    source_branch: str,
    run_id: str,
    command_trace_path: Path,
    timeout_seconds: int,
) -> list[CommandResult]:
    if not source_branch:
        return []
    models_repo = repo / "third_party" / "tt_forge_models"
    tmp_dir = ensure_dir(repo / ".tmp" / "ttxla-profile-discovery")
    safe_branch = slugify(source_branch)
    fetch = run_subprocess(
        command=["git", "fetch", "origin", source_branch],
        cwd=models_repo,
        stdout_path=tmp_dir / f"fetch-{safe_branch}-{run_id}.out",
        stderr_path=tmp_dir / f"fetch-{safe_branch}-{run_id}.err",
        stage="checkout-nvidia-source-branch-fetch",
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
    )
    if not fetch.ok:
        fetch.note = f"failed to fetch tt_forge_models source_branch={source_branch}"
        append_command_trace(command_trace_path, fetch)
        return [fetch]
    checkout = run_subprocess(
        command=["git", "checkout", "--force", "FETCH_HEAD"],
        cwd=models_repo,
        stdout_path=tmp_dir / f"checkout-{safe_branch}-{run_id}.out",
        stderr_path=tmp_dir / f"checkout-{safe_branch}-{run_id}.err",
        stage="checkout-nvidia-source-branch",
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
    )
    checkout.note = f"checked out tt_forge_models source_branch={source_branch}"
    append_command_trace(command_trace_path, checkout)
    return [fetch, checkout]


def checkout_nvidia_source_branch_for_entries(
    repo: Path,
    entries: list[DiscoveryEntry],
    run_id: str,
    command_trace_path: Path,
    timeout_seconds: int,
) -> list[CommandResult]:
    source_branch = source_branch_for_entries(entries)
    if not source_branch:
        return []
    return checkout_nvidia_source_branch(
        repo=repo,
        source_branch=source_branch,
        run_id=run_id,
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
    )


def collect_nvidia_entries_by_source_branch(
    repo: Path,
    run_id: str,
    python_bin: str,
    command_trace_path: Path,
    timeout_seconds: int,
    candidate_entries: list[DiscoveryEntry],
) -> tuple[list[DiscoveryEntry], list[DiscoveryEntry], int, int]:
    selected_entries: list[DiscoveryEntry] = []
    collected_entries: list[DiscoveryEntry] = []
    failed_checkouts = 0
    failed_collections = 0
    branch_groups: dict[str, list[DiscoveryEntry]] = defaultdict(list)
    for entry in candidate_entries:
        branch_groups[entry.source_branch].append(entry)
    for branch_index, branch_entries in enumerate(branch_groups.values(), 1):
        branch_run_id = f"{run_id}-branch-{branch_index:04d}"
        checkout_results = checkout_nvidia_source_branch_for_entries(
            repo=repo,
            entries=branch_entries,
            run_id=branch_run_id,
            command_trace_path=command_trace_path,
            timeout_seconds=timeout_seconds,
        )
        if any(not result.ok for result in checkout_results):
            failed_checkouts += 1
            continue
        branch_collected, branch_result = collect_runner_models_for_nvidia_cohort(
            repo=repo,
            run_id=branch_run_id,
            python_bin=python_bin,
            command_trace_path=command_trace_path,
            timeout_seconds=timeout_seconds,
        )
        if not branch_result.ok:
            failed_collections += 1
            continue
        branch_collected_nodeids = {entry.nodeid for entry in branch_collected}
        selected_entries.extend(
            entry
            for entry in branch_entries
            if entry.nodeid in branch_collected_nodeids
        )
        collected_entries.extend(branch_collected)
    return selected_entries, collected_entries, failed_checkouts, failed_collections


def collect_nvidia_entries_from_current_source(
    repo: Path,
    run_id: str,
    python_bin: str,
    command_trace_path: Path,
    timeout_seconds: int,
    candidate_entries: list[DiscoveryEntry],
) -> tuple[list[DiscoveryEntry], list[DiscoveryEntry], CommandResult]:
    collected_entries, discovery_result = collect_runner_models_for_nvidia_cohort(
        repo=repo,
        run_id=run_id,
        python_bin=python_bin,
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
    )
    collected_nodeids = {entry.nodeid for entry in collected_entries}
    selected_entries = [
        entry for entry in candidate_entries if entry.nodeid in collected_nodeids
    ]
    return selected_entries, collected_entries, discovery_result


def nvidia_missing_entries(
    candidate_entries: list[DiscoveryEntry],
    collected_entries: list[DiscoveryEntry],
    validated_collection: bool,
) -> list[DiscoveryEntry]:
    if not validated_collection:
        return []
    collected_nodeids = {entry.nodeid for entry in collected_entries}
    return [
        entry for entry in candidate_entries if entry.nodeid not in collected_nodeids
    ]


def nvidia_collection_payload(discovery_result: CommandResult) -> dict[str, Any]:
    return {
        "command": shell_join(discovery_result.command),
        "returncode": discovery_result.returncode,
        "timed_out": discovery_result.timed_out,
        "stdout": discovery_result.stdout_path,
        "stderr": discovery_result.stderr_path,
        "note": discovery_result.note,
    }


def nvidia_cohort_mapping_payload(
    cohort_path: Path,
    candidate_entries: list[DiscoveryEntry],
    selected_entries: list[DiscoveryEntry],
    collected_entries: list[DiscoveryEntry],
    discovery_result: CommandResult,
    validated_collection: bool,
) -> dict[str, Any]:
    missing_entries = nvidia_missing_entries(
        candidate_entries, collected_entries, validated_collection
    )
    selected_nodeids = {entry.nodeid for entry in selected_entries}
    return {
        "cohort_path": str(cohort_path),
        "validated_collection": validated_collection,
        "collection": nvidia_collection_payload(discovery_result),
        "counts": {
            "candidate_rows": len(candidate_entries),
            "collected_runner_nodes": len(collected_entries),
            "selected_rows": len(selected_entries),
            "missing_rows": len(missing_entries),
        },
        "selected": [asdict(entry) for entry in selected_entries],
        "missing": [asdict(entry) for entry in missing_entries],
        "candidate_nodeids": [entry.nodeid for entry in candidate_entries],
        "selected_nodeids": [entry.nodeid for entry in selected_entries],
        "collected_matching_nodeids": [
            entry.nodeid
            for entry in collected_entries
            if entry.nodeid in selected_nodeids
        ],
    }


def write_nvidia_cohort_mapping(
    run_dir: Path,
    cohort_path: Path,
    candidate_entries: list[DiscoveryEntry],
    selected_entries: list[DiscoveryEntry],
    collected_entries: list[DiscoveryEntry],
    discovery_result: CommandResult,
    validated_collection: bool,
) -> None:
    write_json(
        run_dir / "nvidia-cohort-mapping.json",
        nvidia_cohort_mapping_payload(
            cohort_path,
            candidate_entries,
            selected_entries,
            collected_entries,
            discovery_result,
            validated_collection,
        ),
    )


def select_nvidia_cohort_entries(
    repo: Path,
    run_dir: Path,
    cohort_path: Path,
    run_id: str,
    python_bin: str,
    command_trace_path: Path,
    timeout_seconds: int,
    validate_collection: bool,
    source_branch_checkout: bool,
) -> tuple[list[DiscoveryEntry], CommandResult]:
    candidate_entries = load_nvidia_cohort_entries(cohort_path, run_id)
    if not validate_collection:
        discovery_result = nvidia_cohort_discovery_result(
            repo, cohort_path, candidate_entries
        )
        write_nvidia_cohort_mapping(
            run_dir,
            cohort_path,
            candidate_entries,
            candidate_entries,
            [],
            discovery_result,
            validated_collection=False,
        )
        return candidate_entries, discovery_result

    if source_branch_checkout:
        selected_entries, collected_entries, failed_checkouts, failed_collections = (
            collect_nvidia_entries_by_source_branch(
                repo=repo,
                run_id=run_id,
                python_bin=python_bin,
                command_trace_path=command_trace_path,
                timeout_seconds=timeout_seconds,
                candidate_entries=candidate_entries,
            )
        )
        discovery_result = nvidia_cohort_discovery_result(
            repo, cohort_path, candidate_entries
        )
        if failed_checkouts or failed_collections:
            discovery_result.returncode = 4
        discovery_result.note = (
            f"validated {len(selected_entries)} of {len(candidate_entries)} "
            "NVIDIA/SILICON_PASS cohort rows with per-source-branch "
            f"tt_forge_models checkout; {failed_checkouts} branch checkout "
            f"command(s) failed; {failed_collections} branch collection "
            "command(s) failed"
        )
    else:
        selected_entries, collected_entries, discovery_result = (
            collect_nvidia_entries_from_current_source(
                repo=repo,
                run_id=run_id,
                python_bin=python_bin,
                command_trace_path=command_trace_path,
                timeout_seconds=timeout_seconds,
                candidate_entries=candidate_entries,
            )
        )
        missing_count = len(candidate_entries) - len(selected_entries)
        discovery_result.note = (
            f"validated {len(selected_entries)} of {len(candidate_entries)} "
            f"NVIDIA/SILICON_PASS cohort rows against TT runner collection; "
            f"{missing_count} rows were not collected"
        )
    write_nvidia_cohort_mapping(
        run_dir,
        cohort_path,
        candidate_entries,
        selected_entries,
        collected_entries,
        discovery_result,
        validated_collection=True,
    )
    return selected_entries, discovery_result


def discover_models(
    repo: Path,
    run_id: str,
    python_bin: str,
    command_trace_path: Path,
    benchmark_paths: Optional[list[Path]] = None,
    timeout_seconds: int = 900,
) -> tuple[list[DiscoveryEntry], CommandResult]:
    files = benchmark_paths or benchmark_files(repo)
    command = collect_command(python_bin, files)
    tmp_dir = ensure_dir(repo / ".tmp" / "ttxla-profile-discovery")
    stdout_path = tmp_dir / f"collect-{run_id}.out"
    stderr_path = tmp_dir / f"collect-{run_id}.err"
    result = run_subprocess(
        command=command,
        cwd=repo,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stage="discover",
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
        env=repo_subprocess_environment(repo),
    )
    collected = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    entries = parse_collect_output(collected, run_id)
    return entries, result


def select_discovery_entries(
    entries: list[DiscoveryEntry],
    nodeid_filters: Iterable[str],
    max_models: int,
) -> list[DiscoveryEntry]:
    filters = [value for value in (item.strip() for item in nodeid_filters) if value]
    selected = entries
    if filters:
        selected = [
            entry
            for entry in selected
            if any(pattern in entry.nodeid for pattern in filters)
        ]
    if max_models > 0:
        selected = selected[:max_models]
    return selected


def preflight_command_for_tool(
    tool_name: str,
    python_bin: str,
    pytest_command: str,
    tracy_bin: str,
    tt_perf_report_bin: str,
) -> list[str]:
    if tool_name == "pytest":
        if pytest_command == "pytest":
            return [python_bin, "-m", "pytest", "--version"]
        return [*split_command_expr(pytest_command), "--version"]
    if tool_name == "tracy":
        return [*split_command_expr(tracy_bin), "--help"]
    if tool_name == "tt-perf-report":
        return [*split_command_expr(tt_perf_report_bin), "--help"]
    raise ValueError(f"unknown preflight tool: {tool_name}")


def run_tool_readiness_checks(
    repo: Path,
    run_dir: Path,
    command_trace_path: Path,
    python_bin: str,
    pytest_command: str,
    tracy_bin: str,
    tt_perf_report_bin: str,
    timeout_seconds: int = 30,
) -> list[CommandResult]:
    readiness_dir = ensure_dir(run_dir / "readiness")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(readiness_dir / "matplotlib-cache"))
    results = []
    for tool_name in ("pytest", "tracy", "tt-perf-report"):
        command = preflight_command_for_tool(
            tool_name, python_bin, pytest_command, tracy_bin, tt_perf_report_bin
        )
        result = run_subprocess(
            command=command,
            cwd=repo,
            stdout_path=readiness_dir / f"{slugify(tool_name)}.out",
            stderr_path=readiness_dir / f"{slugify(tool_name)}.err",
            stage=f"readiness-{tool_name}",
            command_trace_path=command_trace_path,
            timeout_seconds=timeout_seconds,
            env=env,
        )
        results.append(result)
    return results


def readiness_summary(results: list[CommandResult]) -> dict[str, Any]:
    records = []
    for result in results:
        records.append(
            {
                "stage": result.stage,
                "command": shell_join(result.command),
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "ok": result.ok,
                "stdout": result.stdout_path,
                "stderr": result.stderr_path,
                "note": result.note,
            }
        )
    failed = [record for record in records if not record["ok"]]
    return {
        "ok": not failed,
        "failed": failed,
        "checks": records,
    }


def write_readiness_blocker_artifacts(
    run_dir: Path, environment: dict[str, Any], results: list[CommandResult]
) -> None:
    summary = readiness_summary(results)
    manifest = {
        "issue": {
            "number": ISSUE_NUMBER,
            "url": ISSUE_URL,
        },
        "prd": PRD_ID,
        "run": {
            "run_id": run_dir.name,
            "created_at": now_iso(),
            "completed_at": now_iso(),
            "run_dir": str(run_dir),
            "repo_root": environment["repo_root"],
            "status": "environment_blocked",
        },
        "summary": {
            "models": 0,
            "slow_ops": 0,
            "taxonomy": {"environment_failure": 1},
            "readiness": summary,
        },
        "artifacts": {
            "environment": str(run_dir / "environment.json"),
            "readiness": str(run_dir / "readiness"),
            "command_trace": str(run_dir / "command-trace.jsonl"),
        },
    }
    environment["readiness"] = summary
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "model-manifest.json", {"models": []})


def capture_environment(
    repo: Path, tracy_bin: str, tt_perf_report_bin: str, pytest_command: str
) -> dict[str, Any]:
    git_branch = run_simple_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    git_sha = run_simple_git(repo, ["rev-parse", "HEAD"])
    git_status = run_simple_git(repo, ["status", "--short", "--branch"])
    return {
        "captured_at": now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "repo_root": str(repo),
        "git": {
            "branch": git_branch,
            "sha": git_sha,
            "status": git_status,
        },
        "tools": {
            "pytest": pytest_command,
            "tracy": tracy_bin or "",
            "tt_perf_report": tt_perf_report_bin or "",
        },
    }


def run_simple_git(repo: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )
        return (result.stdout or result.stderr or "").strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def copy_tree(source: Path, target: Path) -> list[Path]:
    copied: list[Path] = []
    if not source.exists():
        return copied
    ensure_dir(target)
    for item in source.rglob("*"):
        if item.is_dir():
            continue
        relative = item.relative_to(source)
        dest = target / relative
        ensure_dir(dest.parent)
        shutil.copy2(item, dest)
        copied.append(dest)
    return copied


def find_latest_csv(
    root: Path, patterns: Iterable[str] = PERF_REPORT_CSV_PATTERNS
) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = []
    for pattern in patterns:
        candidates.extend(path for path in root.rglob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def csv_has_columns(csv_path: Path, required_columns: Iterable[str]) -> bool:
    try:
        with csv_path.open(
            "r", encoding="utf-8", errors="replace", newline=""
        ) as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except OSError:
        return False
    normalized = {column.strip() for column in header}
    return all(column in normalized for column in required_columns)


def find_latest_tt_perf_report_csv(root: Path) -> Optional[Path]:
    csv_path = find_latest_csv(root, TT_PERF_REPORT_CSV_PATTERNS)
    if csv_path and csv_has_columns(csv_path, TT_PERF_REPORT_REQUIRED_COLUMNS):
        return csv_path
    return None


def find_latest_tracy_report(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [path for path in root.rglob("*.tracy") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_duration_value(value: str, key_name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except ValueError:
        return None
    lowered = key_name.lower()
    if lowered.endswith("_ns") or "[ns]" in lowered:
        return numeric / 1000.0
    if lowered.endswith("_ms") or "[ms]" in lowered:
        return numeric * 1000.0
    return numeric


def pick_first(row: dict[str, str], aliases: Iterable[str]) -> str:
    normalized = {
        key.lower().strip(): value for key, value in row.items() if key is not None
    }
    for alias in aliases:
        value = normalized.get(alias.lower())
        if value not in (None, ""):
            return value
    return ""


def pick_duration(row: dict[str, str]) -> tuple[str, Optional[float]]:
    for alias in CSV_DURATION_ALIASES:
        candidate = row.get(alias)
        if candidate in (None, ""):
            continue
        duration_value = parse_duration_value(candidate, alias)
        if duration_value is not None:
            return alias, duration_value
    return "", None


def parse_perf_csv(csv_path: Path, model_name: str, model_slug: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_duration_us = 0.0
    op_type_totals: dict[str, float] = defaultdict(float)
    op_name_totals: dict[str, float] = defaultdict(float)

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            op_name = pick_first(row, CSV_OP_ALIASES) or "unknown-op"
            op_type = pick_first(row, CSV_OP_TYPE_ALIASES) or "unknown-type"
            duration_raw_key, duration_value = pick_duration(row)
            if duration_value is None:
                continue
            total_duration_us += duration_value
            op_type_totals[op_type] += duration_value
            op_name_totals[op_name] += duration_value
            rows.append(
                {
                    "model": pick_first(row, CSV_MODEL_ALIASES) or model_name,
                    "model_slug": model_slug,
                    "op_name": op_name,
                    "op_type": op_type,
                    "duration_us": duration_value,
                    "duration_source": duration_raw_key,
                    "raw": row,
                }
            )

    rows.sort(key=lambda item: item["duration_us"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    return {
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "total_duration_us": total_duration_us,
            "op_type_totals": dict(
                sorted(op_type_totals.items(), key=lambda item: item[1], reverse=True)
            ),
            "op_name_totals": dict(
                sorted(op_name_totals.items(), key=lambda item: item[1], reverse=True)
            ),
        },
    }


def text_has_hint(text: str, hints: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def text_has_skip_signal(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"=+.*\bskipped\b.*=+", lowered)
        or re.search(r"\b\d+\s+skipped\b", lowered)
        or re.search(r"(^|\n)\s*skipped\b", lowered)
        or re.search(r"(^|\n)\s*xfailed\b", lowered)
        or re.search(r"(^|\n)\s*xfail\b", lowered)
    )


def text_has_pytest_pass_signal(text: str) -> bool:
    return bool(PYTEST_PASS_PATTERN.search(text))


def infer_profile_status(returncode: Optional[int], timed_out: bool) -> str:
    if timed_out:
        return "pending"
    if returncode == 0:
        return "passed"
    if returncode is None:
        return "unknown"
    return "failed"


def infer_model_status(
    returncode: Optional[int],
    timed_out: bool,
    text: str,
    benchmark_json: dict[str, Any],
) -> str:
    if text_has_skip_signal(text):
        return "skipped"
    if returncode == 0 and benchmark_json:
        return "passed"
    if returncode == 0 and text_has_pytest_pass_signal(text):
        return "passed"
    if benchmark_json and text_has_pytest_pass_signal(text):
        return "passed"
    if timed_out:
        return "pending"
    if text_has_hint(text, PIPELINE_ERROR_HINTS):
        return RUN_STATUS_UNKNOWN
    if text_has_hint(text, ENVIRONMENT_FAILURE_HINTS):
        return RUN_STATUS_NOT_RUN
    if text_has_hint(text, MODEL_FAILURE_HINTS):
        return "failed"
    if returncode == 0:
        return "passed"
    if returncode not in (0, None):
        return "failed" if benchmark_json else "unknown"
    return "unknown"


def infer_taxonomy_from_statuses(
    profile_status: str,
    model_status: str,
    text: str,
    benchmark_json: dict[str, Any],
    perf_report_ok: bool,
) -> tuple[str, str]:
    if profile_status == "pending" or model_status == "pending":
        if benchmark_json:
            return taxonomy_for_pipeline_fallback(benchmark_json)
        return TAXONOMY_NOT_STARTED, "timed out before reaching a terminal state"
    if model_status == "skipped":
        return TAXONOMY_SKIPPED_WITH_REASON, "benchmark entry was skipped"
    if model_status == RUN_STATUS_NOT_RUN or (
        profile_status != "passed" and text_has_hint(text, ENVIRONMENT_FAILURE_HINTS)
    ):
        return (
            TAXONOMY_ENVIRONMENT_FAILURE,
            "environment or dependency issue blocked profiling",
        )
    if model_status == "failed":
        return taxonomy_for_model_failure(text)
    if profile_status == "passed" and model_status == "passed":
        return taxonomy_for_successful_model(perf_report_ok)
    return taxonomy_for_pipeline_fallback(benchmark_json)


def taxonomy_for_model_failure(text: str) -> tuple[str, str]:
    lowered = text.lower()
    if any(
        term in lowered for term in ("pcc comparison failed", "accuracy", "validation")
    ):
        return TAXONOMY_VALIDATED_FAIL, "model compiled but validation failed"
    return (
        TAXONOMY_MODEL_FAILURE,
        "model or runtime behavior failed after environment preconditions passed",
    )


def taxonomy_for_successful_model(perf_report_ok: bool) -> tuple[str, str]:
    if perf_report_ok:
        return (
            TAXONOMY_VALIDATED_PASS,
            "profiling run completed and perf report was produced",
        )
    return (
        TAXONOMY_PIPELINE_ERROR,
        "benchmark completed but perf report could not be produced",
    )


def taxonomy_for_pipeline_fallback(
    benchmark_json: dict[str, Any],
) -> tuple[str, str]:
    if benchmark_json:
        return (
            TAXONOMY_PIPELINE_ERROR,
            "pipeline or artifact stage failed after benchmark output was produced",
        )
    return (
        TAXONOMY_PIPELINE_ERROR,
        "pipeline did not produce a terminal profiling artifact",
    )


def infer_taxonomy(
    returncode: Optional[int],
    timed_out: bool,
    text: str,
    benchmark_json: dict[str, Any],
    perf_report_ok: bool,
) -> tuple[str, str]:
    profile_status = infer_profile_status(returncode, timed_out)
    model_status = infer_model_status(returncode, timed_out, text, benchmark_json)
    return infer_taxonomy_from_statuses(
        profile_status, model_status, text, benchmark_json, perf_report_ok
    )


def terminal_state_for_taxonomy(taxonomy: str) -> str:
    mapping = {
        TAXONOMY_VALIDATED_PASS: TERMINAL_STATE_PASSED,
        TAXONOMY_VALIDATED_FAIL: TERMINAL_STATE_FAILED,
        TAXONOMY_MODEL_FAILURE: TERMINAL_STATE_FAILED,
        TAXONOMY_ENVIRONMENT_FAILURE: TERMINAL_STATE_BLOCKED,
        TAXONOMY_PIPELINE_ERROR: TERMINAL_STATE_BLOCKED,
        TAXONOMY_NOT_RUN: TERMINAL_STATE_BLOCKED,
        TAXONOMY_NOT_STARTED: TERMINAL_STATE_BLOCKED,
        TAXONOMY_SKIPPED_WITH_REASON: TERMINAL_STATE_SKIPPED,
        TAXONOMY_COMPILED_ONLY: TERMINAL_STATE_PARTIAL,
    }
    return mapping.get(taxonomy, TERMINAL_STATE_BLOCKED)


def next_action_for_taxonomy(taxonomy: str) -> str:
    mapping = {
        TAXONOMY_VALIDATED_PASS: "Review dashboard rankings and choose the next optimization target.",
        TAXONOMY_VALIDATED_FAIL: "Fix the validation mismatch, then rerun the model profile.",
        TAXONOMY_COMPILED_ONLY: "Collect the missing validation or perf data, then rerun.",
        TAXONOMY_MODEL_FAILURE: "Resolve the model/runtime failure, then rerun the profile.",
        TAXONOMY_ENVIRONMENT_FAILURE: "Repair the missing dependency or host setup, then rerun.",
        TAXONOMY_PIPELINE_ERROR: "Repair the pipeline or artifact stage, then rerun the profile.",
        TAXONOMY_NOT_RUN: "Run the model profile or record the owner for the non-run decision.",
        TAXONOMY_NOT_STARTED: "Run the model profile or collect the missing job state before rerunning.",
        TAXONOMY_SKIPPED_WITH_REASON: "Review the skip reason and owner before promoting the model.",
    }
    return mapping.get(
        taxonomy, "Review the run artifacts and determine the next action."
    )


def extract_benchmark_model_name(payload: dict[str, Any], fallback: str) -> str:
    for key in ("model", "model_info", "model_rawname"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    config = payload.get("config") or {}
    if isinstance(config, dict):
        for key in ("model_info", "display_name"):
            value = config.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return fallback


def load_profile_benchmark_json(paths: ProfilePaths) -> dict[str, Any]:
    benchmark_json = load_json(paths.benchmark_output)
    if benchmark_json:
        return benchmark_json
    report_candidates = sorted(
        paths.perf_dir.glob("report_perf_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for report_path in report_candidates:
        payload = load_json(report_path)
        if payload:
            return payload
    return {}


def stage_result_paths(profile_dir: Path) -> dict[str, str]:
    return {
        "run_log": str(profile_dir / "run.log"),
        "compile_log": str(profile_dir / "compile.log"),
        "benchmark_json": str(profile_dir / "benchmark.json"),
        "ir_dir": str(profile_dir / "ir"),
        "perf_dir": str(profile_dir / "perf-report"),
        "tracy_dir": str(profile_dir / "tracy"),
        "perf_csv": str(profile_dir / "perf-report" / "tt-perf-report-input.csv"),
        "tt_perf_report": str(profile_dir / "perf-report" / "tt-perf-report.txt"),
        "slow_ops": str(profile_dir / "slow-ops.json"),
    }


def write_unprofiled_model_status(
    entry: DiscoveryEntry,
    repo: Path,
    run_dir: Path,
    taxonomy: str,
    reason: str,
    next_action: str = "",
) -> dict[str, Any]:
    profile_dir = ensure_dir(run_dir / "profiles" / entry.artifact_slug)
    ensure_dir(profile_dir / "perf-report")
    ensure_dir(profile_dir / "ir")
    ensure_dir(profile_dir / "tracy")
    slow_ops_path = profile_dir / "slow-ops.json"
    terminal_state = terminal_state_for_taxonomy(taxonomy)
    if taxonomy in (TAXONOMY_NOT_RUN, TAXONOMY_NOT_STARTED):
        profile_status = RUN_STATUS_NOT_RUN
        model_status = RUN_STATUS_NOT_RUN
    else:
        profile_status = RUN_STATUS_BLOCKED
        model_status = RUN_STATUS_UNKNOWN
    status_payload = {
        "issue": {
            "number": ISSUE_NUMBER,
            "url": ISSUE_URL,
        },
        "prd": PRD_ID,
        "run": {
            "run_id": entry.run_identity,
            "created_at": now_iso(),
        },
        "model": {
            "run_identity": entry.run_identity,
            "nodeid": entry.nodeid,
            "source_path": entry.source_path,
            "test_name": entry.test_name,
            "benchmark_family": entry.benchmark_family,
            "model_identity": entry.model_identity,
            "artifact_slug": entry.artifact_slug,
        },
        "environment": {
            "repo_root": str(repo),
            "hostname": socket.gethostname(),
            "python": sys.version,
            "python_executable": sys.executable,
            "trace_enabled": False,
        },
        "commands": {
            "profile": "",
            "tt_perf_report": "",
        },
        "artifacts": {
            **stage_result_paths(profile_dir),
            "ir_source": "",
            "copied_ir_count": 0,
            "perf_csv_source": "",
            "max_raw_artifact_bytes": 0,
            "pruned_raw_artifacts": [],
        },
        "stages": {
            "profile": {
                "state": profile_status,
                "returncode": None,
                "timed_out": False,
                "note": reason,
                "stdout": str(profile_dir / "run.log"),
                "stderr": str(profile_dir / "compile.log"),
            },
            "ir": {
                "state": "missing",
                "source": "",
                "count": 0,
            },
            "tt_perf_report": {
                "state": "blocked",
                "returncode": None,
                "note": reason,
            },
        },
        "profile_status": profile_status,
        "model_status": model_status,
        "tool_status": {
            "tt_perf_report": "blocked",
            "ir": "missing",
        },
        "taxonomy": taxonomy,
        "terminal_state": terminal_state,
        "reason": reason,
        "next_action": next_action or next_action_for_taxonomy(taxonomy),
        "status_path": str(profile_dir / "status.json"),
        "benchmark": {},
        "slow_ops": str(slow_ops_path),
        "verification": {
            "verified": False,
            "state": "pending",
        },
    }
    slow_ops_payload = {
        "model": entry.model_identity,
        "nodeid": entry.nodeid,
        "source_path": entry.source_path,
        "profile_dir": str(profile_dir),
        "ir_dir": str(profile_dir / "ir"),
        "perf_csv": "",
        "tt_perf_report": str(profile_dir / "perf-report" / "tt-perf-report.txt"),
        "rows": [],
        "summary": {
            "row_count": 0,
            "total_duration_us": 0.0,
            "op_type_totals": {},
            "op_name_totals": {},
        },
    }
    write_json(slow_ops_path, slow_ops_payload)
    write_json(profile_dir / "status.json", status_payload)
    return status_payload


def terminalize_missing_model_statuses(
    run_dir: Path,
    entries: list[DiscoveryEntry],
    repo: Path,
    reason: str,
    taxonomy: str = TAXONOMY_NOT_STARTED,
) -> list[dict[str, Any]]:
    created = []
    for entry in entries:
        status_path = run_dir / "profiles" / entry.artifact_slug / "status.json"
        if status_path.exists():
            continue
        created.append(
            write_unprofiled_model_status(
                entry=entry,
                repo=repo,
                run_dir=run_dir,
                taxonomy=taxonomy,
                reason=reason,
            )
        )
    return created


def load_discovery_entries_from_manifest(run_dir: Path) -> list[DiscoveryEntry]:
    payload = load_json(run_dir / "model-manifest.json")
    entries = []
    for row in payload.get("models", []) or []:
        try:
            entries.append(
                DiscoveryEntry(
                    run_identity=str(row["run_identity"]),
                    nodeid=str(row["nodeid"]),
                    source_path=str(row["source_path"]),
                    test_name=str(row["test_name"]),
                    benchmark_family=str(row["benchmark_family"]),
                    model_identity=str(row["model_identity"]),
                    artifact_slug=str(row["artifact_slug"]),
                )
            )
        except KeyError:
            continue
    return entries


def manifest_run_value(manifest: dict[str, Any], key: str, default: str) -> str:
    run = manifest.get("run") or {}
    return str(run.get(key) or default)


def discovery_result_from_manifest(run_dir: Path) -> CommandResult:
    manifest = load_json(run_dir / "manifest.json")
    discovery = manifest.get("discovery") or {}
    created_at = manifest_run_value(manifest, "created_at", now_iso())
    return CommandResult(
        stage="discover",
        command=shlex.split(str(manifest.get("command") or "")),
        cwd=manifest_run_value(manifest, "repo_root", str(run_dir)),
        returncode=discovery.get("returncode"),
        timed_out=bool(discovery.get("timed_out", False)),
        start_time=created_at,
        end_time=created_at,
        duration_seconds=0.0,
        stdout_path="",
        stderr_path="",
        note=str(discovery.get("note") or ""),
    )


def finalize_partial_run_from_manifest(
    run_dir: Path,
    repo: Path,
    environment: dict[str, Any],
    reason: str,
) -> bool:
    entries = load_discovery_entries_from_manifest(run_dir)
    if not entries:
        return False
    terminalize_missing_model_statuses(
        run_dir=run_dir,
        entries=entries,
        repo=repo,
        reason=reason,
    )
    statuses = load_model_statuses(run_dir)
    slow_ops = load_slow_ops(run_dir)
    write_artifacts(
        run_dir=run_dir,
        environment=environment,
        discovery_result=discovery_result_from_manifest(run_dir),
        entries=entries,
        statuses=statuses,
        slow_ops=slow_ops,
    )
    update_manifest_summary(run_dir, summarize_run(run_dir, statuses, slow_ops))
    return True


def profile_paths(run_dir: Path, entry: DiscoveryEntry) -> ProfilePaths:
    profile_dir = ensure_dir(run_dir / "profiles" / entry.artifact_slug)
    perf_dir = ensure_dir(profile_dir / "perf-report")
    return ProfilePaths(
        profile_dir=profile_dir,
        perf_dir=perf_dir,
        ir_dir=ensure_dir(profile_dir / "ir"),
        trace_dir=ensure_dir(profile_dir / "tracy"),
        benchmark_output=profile_dir / "benchmark.json",
        run_log=profile_dir / "run.log",
        compile_log=profile_dir / "compile.log",
        perf_output=perf_dir / "tt-perf-report.txt",
        perf_input=perf_dir / "tt-perf-report-input.csv",
        slow_ops_path=profile_dir / "slow-ops.json",
    )


def profile_environment(
    repo: Path, entry: DiscoveryEntry, profile_dir: Path
) -> dict[str, str]:
    env = repo_subprocess_environment(repo)
    home_dir = ensure_dir(profile_dir / ".home")
    cache_dir = ensure_dir(home_dir / ".cache")
    env.setdefault("TTMLIR_ENABLE_PERF_TRACE", "1")
    env.setdefault("TT_RUNTIME_TRACE_REGION_SIZE", "10000000")
    env["HOME"] = str(home_dir)
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["MPLCONFIGDIR"] = str(ensure_dir(cache_dir / "matplotlib"))
    env["TTXLA_PROFILE_RUN_ID"] = entry.run_identity
    env["TTXLA_PROFILE_NODEID"] = entry.nodeid
    return env


def collect_ir_artifacts(
    ir_dump_root: Path,
    paths: ProfilePaths,
    benchmark_model_name: str,
    entry: DiscoveryEntry,
) -> tuple[Optional[Path], list[Path]]:
    ir_source_candidates = [
        ir_dump_root / benchmark_model_name,
        ir_dump_root / slugify(benchmark_model_name),
        ir_dump_root / entry.model_identity,
        ir_dump_root / slugify(entry.model_identity),
    ]
    ir_source = next(
        (candidate for candidate in ir_source_candidates if candidate.exists()), None
    )
    copied_ir_files = copy_tree(ir_source, paths.ir_dir) if ir_source else []
    return ir_source, copied_ir_files


def run_tt_perf_report(
    repo: Path,
    paths: ProfilePaths,
    tt_perf_report_bin: str,
    command_trace_path: Path,
    timeout_seconds: int,
) -> PerfReportOutcome:
    slow_ops_csv_source = find_latest_csv(paths.trace_dir)
    perf_csv_source = find_latest_tt_perf_report_csv(paths.trace_dir)
    selected_csv_source = perf_csv_source or slow_ops_csv_source
    perf_csv_recorded = ""
    perf_report_command_list: list[str] = []
    if selected_csv_source:
        perf_csv_recorded = str(paths.perf_input)
        shutil.copy2(selected_csv_source, paths.perf_input)

    if not slow_ops_csv_source:
        return PerfReportOutcome(
            result=None,
            ok=False,
            reason="no ops CSV was produced by the Tracy profile",
            command=[],
            csv_source=None,
            csv_recorded=perf_csv_recorded,
        )
    if not perf_csv_source:
        return PerfReportOutcome(
            result=None,
            ok=False,
            reason=(
                "no tt-perf-report-compatible ops CSV was produced by Tracy; "
                "slow-op rows were parsed from the raw device CSV"
            ),
            command=[],
            csv_source=slow_ops_csv_source,
            csv_recorded=perf_csv_recorded,
        )
    if not command_expr_available(tt_perf_report_bin):
        return PerfReportOutcome(
            result=None,
            ok=False,
            reason="tt-perf-report binary was not available",
            command=[],
            csv_source=perf_csv_source,
            csv_recorded=perf_csv_recorded,
        )

    perf_report_command_list = perf_report_command(tt_perf_report_bin, paths.perf_input)
    result = run_subprocess(
        command=perf_report_command_list,
        cwd=paths.profile_dir,
        stdout_path=paths.perf_output,
        stderr_path=paths.perf_dir / "tt-perf-report.err",
        stage="tt-perf-report",
        command_trace_path=command_trace_path,
        timeout_seconds=min(timeout_seconds, 300),
        env=repo_subprocess_environment(repo),
    )
    reason = ""
    if not result.ok:
        reason = (
            safe_read_text(paths.perf_dir / "tt-perf-report.err")
            or "tt-perf-report returned a non-zero exit code"
        )
    return PerfReportOutcome(
        result=result,
        ok=result.ok,
        reason=reason,
        command=perf_report_command_list,
        csv_source=perf_csv_source,
        csv_recorded=perf_csv_recorded,
    )


def write_slow_ops_payload(
    paths: ProfilePaths,
    benchmark_model_name: str,
    entry: DiscoveryEntry,
    perf_outcome: PerfReportOutcome,
) -> None:
    slow_ops_payload: dict[str, Any] = {
        "model": benchmark_model_name,
        "nodeid": entry.nodeid,
        "source_path": entry.source_path,
        "profile_dir": str(paths.profile_dir),
        "ir_dir": str(paths.ir_dir),
        "perf_csv": perf_outcome.csv_recorded,
        "tt_perf_report": str(paths.perf_output),
        "rows": [],
        "summary": {
            "row_count": 0,
            "total_duration_us": 0.0,
            "op_type_totals": {},
            "op_name_totals": {},
        },
    }
    if perf_outcome.csv_source and perf_outcome.csv_source.exists():
        parsed = parse_perf_csv(
            perf_outcome.csv_source, benchmark_model_name, entry.artifact_slug
        )
        slow_ops_payload["rows"] = parsed["rows"]
        slow_ops_payload["summary"] = parsed["summary"]
    write_json(paths.slow_ops_path, slow_ops_payload)


def profile_log_text(paths: ProfilePaths) -> str:
    return "\n".join(
        text
        for text in (
            safe_read_text(paths.run_log),
            safe_read_text(paths.compile_log),
            safe_read_text(paths.perf_output),
        )
        if text
    )


def profile_verification_state(
    terminal_state: str,
    profile_status: str,
    model_status: str,
    perf_report_ok: bool,
    combined_text: str,
) -> dict[str, Any]:
    verified = (
        terminal_state == "passed"
        and profile_status == "passed"
        and model_status == "passed"
        and perf_report_ok
    )
    return {
        "verified": verified,
        "state": "verified" if verified else "pending",
    }


def profile_stage_payload(
    profile_status: str, profile_result: CommandResult, paths: ProfilePaths
) -> dict[str, Any]:
    return {
        "state": profile_status,
        "returncode": profile_result.returncode,
        "timed_out": profile_result.timed_out,
        "note": profile_result.note,
        "stdout": str(paths.run_log),
        "stderr": str(paths.compile_log),
    }


def ir_stage_payload(
    ir_source: Optional[Path], copied_ir_files: list[Path]
) -> dict[str, Any]:
    return {
        "state": "collected" if copied_ir_files else "missing",
        "source": str(ir_source) if ir_source else "",
        "count": len(copied_ir_files),
    }


def perf_report_stage_payload(perf_outcome: PerfReportOutcome) -> dict[str, Any]:
    return {
        "state": "generated" if perf_outcome.ok else "blocked",
        "returncode": getattr(perf_outcome.result, "returncode", None),
        "note": perf_outcome.reason,
    }


def profile_tool_status(
    copied_ir_files: list[Path], perf_outcome: PerfReportOutcome
) -> dict[str, str]:
    return {
        "tt_perf_report": "generated" if perf_outcome.ok else "blocked",
        "ir": "collected" if copied_ir_files else "missing",
    }


def build_profile_status_payload(
    entry: DiscoveryEntry,
    repo: Path,
    paths: ProfilePaths,
    profile_command_list: list[str],
    profile_result: CommandResult,
    benchmark_model_name: str,
    benchmark_json: dict[str, Any],
    ir_source: Optional[Path],
    copied_ir_files: list[Path],
    perf_outcome: PerfReportOutcome,
    combined_text: str,
    max_raw_artifact_bytes: int,
    pruned_raw_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    taxonomy, reason = infer_taxonomy(
        returncode=profile_result.returncode,
        timed_out=profile_result.timed_out,
        text=combined_text,
        benchmark_json=benchmark_json,
        perf_report_ok=perf_outcome.ok,
    )
    profile_status = infer_profile_status(
        profile_result.returncode, profile_result.timed_out
    )
    model_status = infer_model_status(
        profile_result.returncode,
        profile_result.timed_out,
        combined_text,
        benchmark_json,
    )
    terminal_state = terminal_state_for_taxonomy(taxonomy)
    return {
        "issue": {
            "number": ISSUE_NUMBER,
            "url": ISSUE_URL,
        },
        "prd": PRD_ID,
        "run": {
            "run_id": entry.run_identity,
            "created_at": now_iso(),
        },
        "model": {
            "run_identity": entry.run_identity,
            "nodeid": entry.nodeid,
            "source_path": entry.source_path,
            "test_name": entry.test_name,
            "benchmark_family": entry.benchmark_family,
            "model_identity": benchmark_model_name,
            "artifact_slug": entry.artifact_slug,
        },
        "environment": {
            "repo_root": str(repo),
            "hostname": socket.gethostname(),
            "python": sys.version,
            "python_executable": sys.executable,
            "trace_enabled": True,
        },
        "commands": {
            "profile": shell_join(profile_command_list),
            "tt_perf_report": (
                shell_join(perf_outcome.command) if perf_outcome.command else ""
            ),
        },
        "artifacts": {
            **stage_result_paths(paths.profile_dir),
            "ir_source": str(ir_source) if ir_source else "",
            "copied_ir_count": len(copied_ir_files),
            "perf_csv_source": (
                str(perf_outcome.csv_source) if perf_outcome.csv_source else ""
            ),
            "max_raw_artifact_bytes": max_raw_artifact_bytes,
            "pruned_raw_artifacts": pruned_raw_artifacts,
        },
        "stages": {
            "profile": profile_stage_payload(profile_status, profile_result, paths),
            "ir": ir_stage_payload(ir_source, copied_ir_files),
            "tt_perf_report": perf_report_stage_payload(perf_outcome),
        },
        "profile_status": profile_status,
        "model_status": model_status,
        "tool_status": profile_tool_status(copied_ir_files, perf_outcome),
        "taxonomy": taxonomy,
        "terminal_state": terminal_state,
        "reason": reason,
        "next_action": next_action_for_taxonomy(taxonomy),
        "status_path": str(paths.profile_dir / "status.json"),
        "benchmark": benchmark_json,
        "slow_ops": str(paths.slow_ops_path),
        "verification": profile_verification_state(
            terminal_state,
            profile_status,
            model_status,
            perf_outcome.ok,
            combined_text,
        ),
    }


def profile_one_model(
    entry: DiscoveryEntry,
    repo: Path,
    run_dir: Path,
    pytest_command: str,
    tracy_bin: str,
    tt_perf_report_bin: str,
    command_trace_path: Path,
    timeout_seconds: int,
    benchmark_kwargs: dict[str, int],
    max_raw_artifact_bytes: int,
    ir_dump_root: Path,
) -> dict[str, Any]:
    paths = profile_paths(run_dir, entry)
    clean_module_cache(repo)
    profile_command_list = profile_command(
        tracy_bin=tracy_bin,
        pytest_command=pytest_command,
        nodeid=entry.nodeid,
        profile_dir=paths.profile_dir,
        benchmark_output=paths.benchmark_output,
        ir_dump_root=ir_dump_root,
        benchmark_args=benchmark_args_for_entry(entry, benchmark_kwargs),
    )

    profile_result = run_subprocess(
        command=profile_command_list,
        cwd=repo,
        stdout_path=paths.run_log,
        stderr_path=paths.compile_log,
        stage="profile",
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
        env=profile_environment(repo, entry, paths.profile_dir),
    )
    pruned_raw_artifacts = prune_large_raw_artifacts(
        paths.profile_dir, max_raw_artifact_bytes
    )

    benchmark_json = load_json(paths.benchmark_output)
    benchmark_model_name = extract_benchmark_model_name(
        benchmark_json, entry.model_identity
    )
    ir_source, copied_ir_files = collect_ir_artifacts(
        ir_dump_root, paths, benchmark_model_name, entry
    )
    perf_outcome = run_tt_perf_report(
        repo, paths, tt_perf_report_bin, command_trace_path, timeout_seconds
    )
    write_slow_ops_payload(paths, benchmark_model_name, entry, perf_outcome)
    combined_text = profile_log_text(paths)
    status_payload = build_profile_status_payload(
        entry=entry,
        repo=repo,
        paths=paths,
        profile_command_list=profile_command_list,
        profile_result=profile_result,
        benchmark_model_name=benchmark_model_name,
        benchmark_json=benchmark_json,
        ir_source=ir_source,
        copied_ir_files=copied_ir_files,
        perf_outcome=perf_outcome,
        combined_text=combined_text,
        max_raw_artifact_bytes=max_raw_artifact_bytes,
        pruned_raw_artifacts=pruned_raw_artifacts,
    )

    write_json(paths.profile_dir / "status.json", status_payload)
    return status_payload


def clean_module_cache(repo: Path) -> None:
    modules_dir = repo / "modules"
    if modules_dir.exists():
        shutil.rmtree(modules_dir)


def discover_artifacts(
    run_dir: Path,
    entries: list[DiscoveryEntry],
    discovery_result: CommandResult,
    environment: dict[str, Any],
) -> None:
    manifest = {
        "issue": {
            "number": ISSUE_NUMBER,
            "url": ISSUE_URL,
        },
        "prd": PRD_ID,
        "run": {
            "run_id": run_dir.name,
            "created_at": now_iso(),
            "run_dir": str(run_dir),
            "repo_root": environment["repo_root"],
            "status": "discovered",
        },
        "command": shell_join(discovery_result.command),
        "discovery": {
            "returncode": discovery_result.returncode,
            "timed_out": discovery_result.timed_out,
            "note": discovery_result.note,
        },
        "counts": {
            "discovered_models": len(entries),
        },
        "benchmark_files": [entry.source_path for entry in entries],
        "artifacts": {
            "environment": str(run_dir / "environment.json"),
            "model_manifest": str(run_dir / "model-manifest.json"),
            "command_trace": str(run_dir / "command-trace.jsonl"),
        },
    }
    nvidia_mapping_path = run_dir / "nvidia-cohort-mapping.json"
    if nvidia_mapping_path.exists():
        manifest["artifacts"]["nvidia_cohort_mapping"] = str(nvidia_mapping_path)
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "environment.json", environment)
    write_json(
        run_dir / "model-manifest.json",
        {"models": [asdict(entry) for entry in entries]},
    )


def update_manifest_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path)
    manifest.setdefault("run", {})
    manifest["run"]["completed_at"] = now_iso()
    manifest["summary"] = summary
    write_json(manifest_path, manifest)


def load_model_statuses(run_dir: Path) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    profiles_dir = run_dir / "profiles"
    if not profiles_dir.exists():
        return statuses
    for status_file in sorted(profiles_dir.glob("*/status.json")):
        status = load_json(status_file)
        status.setdefault("status_path", str(status_file))
        statuses.append(status)
    return statuses


def load_slow_ops(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    profiles_dir = run_dir / "profiles"
    if not profiles_dir.exists():
        return records
    for slow_ops_file in sorted(profiles_dir.glob("*/slow-ops.json")):
        payload = load_json(slow_ops_file)
        model_status = load_json(slow_ops_file.parent / "status.json")
        for row in payload.get("rows", []) or []:
            record = dict(row)
            record["model_identity"] = payload.get(
                "model", model_status.get("model", {}).get("model_identity", "unknown")
            )
            record["nodeid"] = payload.get(
                "nodeid", model_status.get("model", {}).get("nodeid", "")
            )
            record["profile_status"] = model_status.get(
                "profile_status", model_status.get("terminal_state", "unknown")
            )
            record["model_status"] = model_status.get("model_status", "unknown")
            record["taxonomy"] = model_status.get("taxonomy", "unknown")
            record["profile_dir"] = str(slow_ops_file.parent)
            record["status_path"] = str(slow_ops_file.parent / "status.json")
            record["ir_dir"] = model_status.get("artifacts", {}).get("ir_dir", "")
            record["perf_report"] = model_status.get("artifacts", {}).get(
                "tt_perf_report", ""
            )
            records.append(record)
    records.sort(key=lambda item: item.get("duration_us", 0.0), reverse=True)
    for index, record in enumerate(records, start=1):
        record["global_rank"] = index
    return records


def collector_job_id(value: str) -> str:
    if not value:
        return DEFAULT_COLLECTOR_JOB_ID
    digits = "".join(char for char in value if char.isdigit())
    return digits or DEFAULT_COLLECTOR_JOB_ID


def clean_superset_perf_reports(export_dir: Path) -> None:
    if not export_dir.exists():
        return
    for path in export_dir.glob("perf_report_ttxla_slow_op_*.json"):
        path.unlink()


def slow_op_perf_report_measurements(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "iteration": 1,
            "step_name": "ttxla_slow_op_profile",
            "step_warm_up_num_iterations": 0,
            "measurement_name": "duration_us",
            "value": float(row.get("duration_us", 0.0)),
            "target": -1,
            "device_power": -1.0,
            "device_temperature": -1.0,
        }
    ]


def slow_op_perf_report_payload(
    run_dir: Path, environment: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    device_target = environment.get("target", {})
    config = {
        "issue": ISSUE_NUMBER,
        "prd": PRD_ID,
        "run_id": run_dir.name,
        "nodeid": row.get("nodeid", ""),
        "source_path": row.get("source_path", ""),
        "op_name": row.get("op_name", ""),
        "op_type": row.get("op_type", ""),
        "rank": row.get("rank"),
        "global_rank": row.get("global_rank"),
        "profile_status": row.get("profile_status", ""),
        "model_status": row.get("model_status", ""),
        "taxonomy": row.get("taxonomy", ""),
        "status_path": row.get("status_path", ""),
        "ir_dir": row.get("ir_dir", ""),
        "perf_report": row.get("perf_report", ""),
        "profile_dir": row.get("profile_dir", ""),
    }
    return {
        "model": row.get("model_identity", row.get("model", "unknown")),
        "model_type": "ttxla_slow_op",
        "run_type": "ttxla_slow_op_profile",
        "config": config,
        "num_layers": None,
        "batch_size": None,
        "precision": "",
        "dataset_name": row.get("op_type", ""),
        "profile_name": "tracy_tt_perf_report",
        "input_sequence_length": None,
        "output_sequence_length": None,
        "image_dimension": "",
        "perf_analysis": True,
        "measurements": slow_op_perf_report_measurements(row),
        "device_info": {
            "device_name": device_target.get("machine", ""),
            "arch": device_target.get("arch", ""),
            "device_count": device_target.get("num_pcie_chips"),
            "mesh_shape": None,
            "device_type": device_target.get("scope", ""),
        },
    }


def write_superset_perf_reports(
    run_dir: Path,
    environment: dict[str, Any],
    slow_ops: list[dict[str, Any]],
    job_id: str,
) -> Path:
    export_dir = ensure_dir(run_dir / "perf_reports" / "slow_ops")
    clean_superset_perf_reports(export_dir)
    suffix = collector_job_id(job_id)
    for index, row in enumerate(slow_ops, start=1):
        payload = slow_op_perf_report_payload(run_dir, environment, row)
        model_slug = slugify(str(payload["model"]))
        op_slug = slugify(str(row.get("op_name", "unknown-op")))
        report_path = (
            export_dir
            / f"perf_report_ttxla_slow_op_{index:05d}_{model_slug}_{op_slug}_{suffix}.json"
        )
        write_json(report_path, payload)
    return export_dir


def aggregate_models(
    statuses: list[dict[str, Any]], slow_ops: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    slow_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in slow_ops:
        slow_by_model[row.get("model_identity", "unknown")].append(row)

    model_rows: list[dict[str, Any]] = []
    for status in statuses:
        model = status.get("model", {})
        model_identity = model.get("model_identity", "unknown")
        rows = slow_by_model.get(model_identity, [])
        total_duration = sum(float(row.get("duration_us", 0.0)) for row in rows)
        top_row = rows[0] if rows else {}
        model_rows.append(
            {
                "model_identity": model_identity,
                "nodeid": model.get("nodeid", ""),
                "source_path": model.get("source_path", ""),
                "profile_status": status.get(
                    "profile_status", status.get("terminal_state", "unknown")
                ),
                "model_status": status.get("model_status", "unknown"),
                "taxonomy": status.get("taxonomy", "unknown"),
                "reason": status.get("reason", ""),
                "next_action": status.get("next_action", ""),
                "ir_dir": status.get("artifacts", {}).get("ir_dir", ""),
                "perf_report": status.get("artifacts", {}).get("tt_perf_report", ""),
                "slow_ops_path": status.get("slow_ops", ""),
                "copied_ir_count": status.get("artifacts", {}).get(
                    "copied_ir_count", 0
                ),
                "perf_report_state": status.get("stages", {})
                .get("tt_perf_report", {})
                .get("state", "unknown"),
                "row_count": len(rows),
                "total_duration_us": total_duration,
                "slowest_op": top_row.get("op_name", ""),
                "slowest_op_type": top_row.get("op_type", ""),
                "slowest_op_duration_us": top_row.get("duration_us", 0.0),
            }
        )
    model_rows.sort(key=lambda item: item["total_duration_us"], reverse=True)
    for index, model in enumerate(model_rows, start=1):
        model["global_rank"] = index
    return model_rows


def aggregate_op_types(slow_ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in slow_ops:
        op_type = row.get("op_type", "unknown-type")
        bucket = grouped.setdefault(
            op_type,
            {
                "op_type": op_type,
                "total_duration_us": 0.0,
                "count": 0,
                "models": set(),
                "top_op": "",
                "top_model": "",
                "top_duration_us": 0.0,
            },
        )
        duration = float(row.get("duration_us", 0.0))
        bucket["total_duration_us"] += duration
        bucket["count"] += 1
        bucket["models"].add(row.get("model_identity", "unknown"))
        if duration > bucket["top_duration_us"]:
            bucket["top_duration_us"] = duration
            bucket["top_op"] = row.get("op_name", "")
            bucket["top_model"] = row.get("model_identity", "")
    rows = []
    for bucket in grouped.values():
        bucket["models"] = len(bucket["models"])
        rows.append(bucket)
    rows.sort(key=lambda item: item["total_duration_us"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def render_links(run_dir: Path, path: str, label: str) -> str:
    if not path:
        return ""
    rel = os.path.relpath(path, run_dir)
    return f'<a href="{html.escape(rel)}">{html.escape(label)}</a>'


def dashboard_filter_values(
    slow_ops: list[dict[str, Any]], models: list[dict[str, Any]], key: str
) -> list[str]:
    return sorted(
        {row.get(key, "unknown") for row in slow_ops}
        | {row.get(key, "unknown") for row in models}
    )


def render_slow_op_row(run_dir: Path, row: dict[str, Any]) -> str:
    return (
        "<tr "
        f'data-model="{html.escape(str(row.get("model_identity", "")))}" '
        f'data-op="{html.escape(str(row.get("op_name", "")))}" '
        f'data-op-type="{html.escape(str(row.get("op_type", "")))}" '
        f'data-status="{html.escape(str(row.get("profile_status", "")))}" '
        f'data-model-status="{html.escape(str(row.get("model_status", "")))}" '
        f'data-taxonomy="{html.escape(str(row.get("taxonomy", "")))}">'
        f"<td>{row.get('global_rank', '')}</td>"
        f"<td>{html.escape(str(row.get('model_identity', '')))}</td>"
        f"<td>{html.escape(str(row.get('op_name', '')))}</td>"
        f"<td>{html.escape(str(row.get('op_type', '')))}</td>"
        f"<td>{float(row.get('duration_us', 0.0)):.2f}</td>"
        f"<td>{html.escape(str(row.get('profile_status', '')))}</td>"
        f"<td>{html.escape(str(row.get('model_status', '')))}</td>"
        f"<td>{html.escape(str(row.get('taxonomy', '')))}</td>"
        f"<td>{render_links(run_dir, str(row.get('status_path', '')), 'status')}</td>"
        f"<td>{render_links(run_dir, str(row.get('ir_dir', '')), 'ir')}</td>"
        f"<td>{render_links(run_dir, str(row.get('perf_report', '')), 'perf')}</td>"
        "</tr>"
    )


def render_model_row(run_dir: Path, row: dict[str, Any]) -> str:
    return (
        "<tr "
        f'data-model="{html.escape(str(row.get("model_identity", "")))}" '
        f'data-status="{html.escape(str(row.get("profile_status", "")))}" '
        f'data-model-status="{html.escape(str(row.get("model_status", "")))}" '
        f'data-taxonomy="{html.escape(str(row.get("taxonomy", "")))}">'
        f"<td>{row.get('global_rank', '')}</td>"
        f"<td>{html.escape(str(row.get('model_identity', '')))}</td>"
        f"<td>{html.escape(str(row.get('profile_status', '')))}</td>"
        f"<td>{html.escape(str(row.get('model_status', '')))}</td>"
        f"<td>{html.escape(str(row.get('taxonomy', '')))}</td>"
        f"<td>{row.get('row_count', 0)}</td>"
        f"<td>{row.get('total_duration_us', 0.0):.2f}</td>"
        f"<td>{html.escape(str(row.get('slowest_op', '')))}</td>"
        f"<td>{html.escape(str(row.get('slowest_op_type', '')))}</td>"
        f"<td>{render_links(run_dir, str(row.get('ir_dir', '')), 'ir')}</td>"
        f"<td>{render_links(run_dir, str(row.get('perf_report', '')), 'perf')}</td>"
        "</tr>"
    )


def render_op_type_row(row: dict[str, Any]) -> str:
    return (
        "<tr "
        f'data-op-type="{html.escape(str(row.get("op_type", "")))}">'
        f"<td>{row.get('rank', '')}</td>"
        f"<td>{html.escape(str(row.get('op_type', '')))}</td>"
        f"<td>{row.get('count', 0)}</td>"
        f"<td>{row.get('models', 0)}</td>"
        f"<td>{row.get('total_duration_us', 0.0):.2f}</td>"
        f"<td>{html.escape(str(row.get('top_model', '')))}</td>"
        f"<td>{html.escape(str(row.get('top_op', '')))}</td>"
        "</tr>"
    )


def dashboard_metric_counts(
    models: list[dict[str, Any]], slow_ops: list[dict[str, Any]]
) -> dict[str, int]:
    return {
        "total_models": len(models),
        "total_ops": len(slow_ops),
        "passed_models": sum(
            1 for row in models if row.get("profile_status") == "passed"
        ),
        "blocked_models": sum(
            1 for row in models if row.get("profile_status") in ("blocked", "pending")
        ),
        "failed_models": sum(
            1 for row in models if row.get("model_status") == "failed"
        ),
    }


def render_dashboard_html(
    run_dir: Path, statuses: list[dict[str, Any]], slow_ops: list[dict[str, Any]]
) -> str:
    models = aggregate_models(statuses, slow_ops)
    op_types = aggregate_op_types(slow_ops)
    all_status_values = dashboard_filter_values(slow_ops, models, "profile_status")
    all_model_statuses = dashboard_filter_values(slow_ops, models, "model_status")
    all_taxonomies = dashboard_filter_values(slow_ops, models, "taxonomy")
    html_rows = [render_slow_op_row(run_dir, row) for row in slow_ops]
    model_rows = [render_model_row(run_dir, row) for row in models]
    op_type_rows = [render_op_type_row(row) for row in op_types]
    metrics = dashboard_metric_counts(models, slow_ops)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TT-XLA profiling dashboard</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #f7f7f8; }}
    header, section {{ padding: 20px 24px; }}
    header {{ background: #111827; color: #f9fafb; }}
    h1, h2, h3 {{ margin: 0 0 12px 0; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 1.6rem; margin-bottom: 4px; }}
    .controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 12px 0 0; }}
    .controls label {{ display: block; font-size: 0.85rem; color: #6b7280; margin-bottom: 6px; }}
    input, select {{ width: 100%; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 8px; background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; }}
    thead {{ background: #f3f4f6; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
    th {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.02em; }}
    tbody tr.hidden {{ display: none; }}
    .table-wrap {{ overflow-x: auto; }}
    .muted {{ color: #6b7280; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .section-note {{ margin-top: 8px; color: #4b5563; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <header>
    <h1>TT-XLA profiling dashboard</h1>
    <div class="muted">Issue #{ISSUE_NUMBER} | {html.escape(PRD_ID)} | run {html.escape(run_dir.name)}</div>
  </header>
  <section>
    <div class="summary">
      <div class="metric"><strong>{metrics["total_models"]}</strong><span>models in scope</span></div>
      <div class="metric"><strong>{metrics["passed_models"]}</strong><span>passed</span></div>
      <div class="metric"><strong>{metrics["blocked_models"]}</strong><span>blocked</span></div>
      <div class="metric"><strong>{metrics["failed_models"]}</strong><span>failed</span></div>
      <div class="metric"><strong>{metrics["total_ops"]}</strong><span>slow-op rows</span></div>
    </div>
    <div class="controls">
      <div><label for="search">Search</label><input id="search" type="search" placeholder="model, op, or path" /></div>
      <div><label for="status">Profile status</label><select id="status"><option value="">All</option>{''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in all_status_values)}</select></div>
      <div><label for="model-status">Model status</label><select id="model-status"><option value="">All</option>{''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in all_model_statuses)}</select></div>
      <div><label for="taxonomy">Taxonomy</label><select id="taxonomy"><option value="">All</option>{''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in all_taxonomies)}</select></div>
    </div>
  </section>
  <section>
    <h2>Global slow operations</h2>
    <p class="section-note">Rows are ranked by duration and carry links back to per-model artifacts.</p>
    <div class="table-wrap">
      <table id="slow-ops">
        <thead>
          <tr><th>#</th><th>Model</th><th>Operation</th><th>Type</th><th>Duration us</th><th>Profile</th><th>Model</th><th>Taxonomy</th><th>Status</th><th>IR</th><th>Perf</th></tr>
        </thead>
        <tbody>
          {''.join(html_rows) or '<tr><td colspan="11">No slow-op data available.</td></tr>'}
        </tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>By model</h2>
    <div class="table-wrap">
      <table id="models">
        <thead>
          <tr><th>#</th><th>Model</th><th>Profile</th><th>Model</th><th>Taxonomy</th><th>Ops</th><th>Total us</th><th>Slowest op</th><th>Type</th><th>IR</th><th>Perf</th></tr>
        </thead>
        <tbody>
          {''.join(model_rows) or '<tr><td colspan="11">No model summary available.</td></tr>'}
        </tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>By op type</h2>
    <div class="table-wrap">
      <table id="op-types">
        <thead>
          <tr><th>#</th><th>Op type</th><th>Count</th><th>Models</th><th>Total us</th><th>Top model</th><th>Top op</th></tr>
        </thead>
        <tbody>
          {''.join(op_type_rows) or '<tr><td colspan="7">No op-type summary available.</td></tr>'}
        </tbody>
      </table>
    </div>
  </section>
  <script>
    const search = document.getElementById('search');
    const status = document.getElementById('status');
    const modelStatus = document.getElementById('model-status');
    const taxonomy = document.getElementById('taxonomy');
    function matches(row) {{
      const text = `${{row.dataset.model || ''}} ${{row.dataset.op || ''}} ${{row.dataset.opType || ''}}`.toLowerCase();
      const needle = search.value.trim().toLowerCase();
      if (needle && !text.includes(needle)) {{
        return false;
      }}
      if (status.value && row.hasAttribute('data-status') && row.dataset.status !== status.value) {{
        return false;
      }}
      if (modelStatus.value && row.hasAttribute('data-model-status') && row.dataset.modelStatus !== modelStatus.value) {{
        return false;
      }}
      if (taxonomy.value && row.hasAttribute('data-taxonomy') && row.dataset.taxonomy !== taxonomy.value) {{
        return false;
      }}
      return true;
    }}
    function applyFilters() {{
      document.querySelectorAll('tbody tr').forEach((row) => {{
        row.classList.toggle('hidden', !matches(row));
      }});
    }}
    search.addEventListener('input', applyFilters);
    status.addEventListener('change', applyFilters);
    modelStatus.addEventListener('change', applyFilters);
    taxonomy.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def render_packet_html(
    run_dir: Path,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    statuses: list[dict[str, Any]],
    slow_ops: list[dict[str, Any]],
) -> str:
    coverage = requirement_coverage(run_dir, manifest, environment, statuses, slow_ops)
    rows = []
    for entry in coverage:
        rows.append(
            f"<tr><td>{html.escape(entry['id'])}</td><td>{html.escape(entry['status'])}</td><td>{html.escape(entry['evidence'])}</td></tr>"
        )
    model_rows = []
    for status in statuses:
        model = status.get("model", {})
        model_rows.append(
            "<tr>"
            f"<td>{html.escape(str(model.get('model_identity', '')))}</td>"
            f"<td>{html.escape(str(status.get('terminal_state', '')))}</td>"
            f"<td>{html.escape(str(status.get('profile_status', '')))}</td>"
            f"<td>{html.escape(str(status.get('model_status', '')))}</td>"
            f"<td>{html.escape(str(status.get('taxonomy', '')))}</td>"
            f"<td>{html.escape(str(status.get('reason', '')))}</td>"
            f"<td>{render_links(run_dir, status.get('artifacts', {}).get('tt_perf_report', ''), 'perf')}</td>"
            f"<td>{render_links(run_dir, status.get('artifacts', {}).get('ir_dir', ''), 'ir')}</td>"
            f"<td>{render_links(run_dir, status.get('slow_ops', ''), 'slow ops')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TT-XLA profiling packet</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #fafafa; }}
    header, section {{ padding: 20px 24px; }}
    header {{ background: #0f172a; color: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <header>
    <h1>TT-XLA profiling evidence packet</h1>
    <div class="muted">Issue #{ISSUE_NUMBER} | {html.escape(PRD_ID)} | run {html.escape(run_dir.name)}</div>
  </header>
  <section>
    <h2>Environment</h2>
    <table>
      <tbody>
        <tr><th>Repo root</th><td>{html.escape(str(environment.get("repo_root", "")))}</td></tr>
        <tr><th>Hostname</th><td>{html.escape(str(environment.get("hostname", "")))}</td></tr>
        <tr><th>Python</th><td>{html.escape(str(environment.get("python", "")))}</td></tr>
        <tr><th>Git SHA</th><td>{html.escape(str(environment.get("git", {}).get("sha", "")))}</td></tr>
        <tr><th>Git branch</th><td>{html.escape(str(environment.get("git", {}).get("branch", "")))}</td></tr>
      </tbody>
    </table>
  </section>
  <section>
    <h2>Requirement coverage</h2>
    <table>
      <thead><tr><th>Requirement</th><th>Status</th><th>Evidence</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
  <section>
    <h2>Models</h2>
    <table>
      <thead><tr><th>Model</th><th>Terminal</th><th>Profile</th><th>Model</th><th>Taxonomy</th><th>Reason</th><th>Perf</th><th>IR</th><th>Slow ops</th></tr></thead>
      <tbody>{''.join(model_rows) or '<tr><td colspan="9">No model status artifacts were generated.</td></tr>'}</tbody>
    </table>
  </section>
</body>
</html>
"""


def render_report_html(
    run_dir: Path,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    statuses: list[dict[str, Any]],
    slow_ops: list[dict[str, Any]],
    dashboard_path: Path,
    packet_path: Path,
) -> str:
    coverage = requirement_coverage(run_dir, manifest, environment, statuses, slow_ops)
    coverage_rows = [render_coverage_row(entry) for entry in coverage]
    counts = report_counts(statuses, slow_ops)
    blocker_rows = [
        render_blocker_row(status)
        for status in statuses
        if status.get("terminal_state") != "passed"
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TT-XLA profiling report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #f8fafc; }}
    header, section {{ padding: 20px 24px; }}
    header {{ background: #111827; color: #f9fafb; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
    .metric {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 1.5rem; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <header>
    <h1>TT-XLA profiling report</h1>
    <div class="muted">Issue #{ISSUE_NUMBER} | {html.escape(PRD_ID)} | run {html.escape(run_dir.name)}</div>
  </header>
  <section>
    <div class="summary">
      <div class="metric"><strong>{counts['models']}</strong><span>models</span></div>
      <div class="metric"><strong>{counts['passed']}</strong><span>passed</span></div>
      <div class="metric"><strong>{counts['failed']}</strong><span>failed</span></div>
      <div class="metric"><strong>{counts['blocked']}</strong><span>blocked</span></div>
      <div class="metric"><strong>{counts['model_failures']}</strong><span>model failures</span></div>
      <div class="metric"><strong>{counts['ops']}</strong><span>slow-op rows</span></div>
    </div>
    <p class="muted">Source packet: <a href="{html.escape(os.path.relpath(packet_path, run_dir))}">claude-report-packet.html</a> | Dashboard: <a href="{html.escape(os.path.relpath(dashboard_path, run_dir))}">dashboard.html</a></p>
  </section>
  <section>
    <h2>Requirement coverage</h2>
    <table>
      <thead><tr><th>Requirement</th><th>Status</th><th>Evidence</th></tr></thead>
      <tbody>{''.join(coverage_rows)}</tbody>
    </table>
  </section>
  <section>
    <h2>Open blockers and next actions</h2>
    <table>
      <thead><tr><th>Model</th><th>Terminal</th><th>Profile</th><th>Model</th><th>Taxonomy</th><th>Next action</th></tr></thead>
      <tbody>{''.join(blocker_rows) or '<tr><td colspan="6">No open blockers.</td></tr>'}</tbody>
    </table>
  </section>
  <section>
    <h2>Environment</h2>
    <table>
      <tbody>
        <tr><th>Repo root</th><td>{html.escape(str(environment.get("repo_root", "")))}</td></tr>
        <tr><th>Hostname</th><td>{html.escape(str(environment.get("hostname", "")))}</td></tr>
        <tr><th>Python</th><td>{html.escape(str(environment.get("python", "")))}</td></tr>
        <tr><th>Git SHA</th><td>{html.escape(str(environment.get("git", {}).get("sha", "")))}</td></tr>
        <tr><th>Git branch</th><td>{html.escape(str(environment.get("git", {}).get("branch", "")))}</td></tr>
      </tbody>
    </table>
  </section>
</body>
</html>
"""


def render_coverage_row(entry: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(entry['id'])}</td>"
        f"<td>{html.escape(entry['status'])}</td>"
        f"<td>{html.escape(entry['evidence'])}</td>"
        "</tr>"
    )


def report_counts(
    statuses: list[dict[str, Any]], slow_ops: list[dict[str, Any]]
) -> dict[str, int]:
    return {
        "models": len(statuses),
        "passed": count_status_value(statuses, "terminal_state", "passed"),
        "failed": count_status_value(statuses, "terminal_state", "failed"),
        "blocked": count_status_value(statuses, "terminal_state", "blocked"),
        "skipped": count_status_value(statuses, "terminal_state", "skipped"),
        "model_failures": count_status_value(statuses, "model_status", "failed"),
        "ops": len(slow_ops),
    }


def count_status_value(statuses: list[dict[str, Any]], key: str, value: str) -> int:
    return sum(1 for status in statuses if status.get(key) == value)


def render_blocker_row(status: dict[str, Any]) -> str:
    model = status.get("model", {})
    return (
        "<tr>"
        f"<td>{html.escape(str(model.get('model_identity', '')))}</td>"
        f"<td>{html.escape(str(status.get('terminal_state', '')))}</td>"
        f"<td>{html.escape(str(status.get('profile_status', '')))}</td>"
        f"<td>{html.escape(str(status.get('model_status', '')))}</td>"
        f"<td>{html.escape(str(status.get('taxonomy', '')))}</td>"
        f"<td>{html.escape(str(status.get('next_action', '')))}</td>"
        "</tr>"
    )


def coverage_entry(
    requirement_index: int,
    status: str,
    evidence: str,
    evidence_paths: Iterable[str],
) -> dict[str, Any]:
    requirement_id, description = REQS[requirement_index]
    return {
        "id": requirement_id,
        "description": description,
        "status": status,
        "evidence": evidence,
        "evidence_paths": list(evidence_paths),
    }


def existing_path_strings(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


def status_stage_count(statuses: list[dict[str, Any]], stage: str, state: str) -> int:
    return sum(
        1
        for status in statuses
        if status.get("stages", {}).get(stage, {}).get("state") == state
    )


def status_artifact_paths(
    statuses: list[dict[str, Any]], artifact_key: str
) -> list[str]:
    return [
        str(Path(str(path)))
        for status in statuses
        if (path := status.get("artifacts", {}).get(artifact_key))
        and path_exists(str(path))
    ]


def dashboard_contains_rankings(dashboard_text: str) -> bool:
    return all(
        marker in dashboard_text
        for marker in ("Global slow operations", 'id="models"', 'id="op-types"')
    )


def dashboard_contains_search_controls(dashboard_text: str) -> bool:
    return all(
        marker in dashboard_text
        for marker in (
            'id="search"',
            'id="status"',
            'id="model-status"',
            'id="taxonomy"',
        )
    )


def model_manifest_has_run_identities(model_manifest: dict[str, Any]) -> bool:
    return all("run_identity" in row for row in model_manifest.get("models", []))


def report_artifacts_have_requirement_coverage(
    packet_path: Path,
    report_path: Path,
) -> bool:
    packet_text = read_text_if_exists(packet_path)
    report_text = read_text_if_exists(report_path)
    return (
        packet_path.exists()
        and report_path.exists()
        and "Requirement coverage" in packet_text
        and "Requirement coverage" in report_text
    )


def requirement_coverage_context(
    run_dir: Path,
    statuses: list[dict[str, Any]],
) -> RequirementCoverageContext:
    model_manifest_path = run_dir / "model-manifest.json"
    model_manifest = load_json(model_manifest_path)
    model_count = len(model_manifest.get("models", []))
    status_count = len(statuses)
    status_paths = [
        Path(str(status.get("status_path", "")))
        for status in statuses
        if status.get("status_path")
    ]
    status_files_exist = status_count > 0 and all(
        path.exists() for path in status_paths
    )
    perf_report_count = status_stage_count(statuses, "tt_perf_report", "generated")
    ir_count = status_stage_count(statuses, "ir", "collected")
    dashboard_path = run_dir / "dashboard.html"
    packet_path = run_dir / "claude-report-packet.html"
    report_path = run_dir / "report.html"
    dashboard_text = read_text_if_exists(dashboard_path)
    superset_export_dir = run_dir / "perf_reports" / "slow_ops"
    superset_export_paths = sorted(superset_export_dir.glob("*.json"))
    return RequirementCoverageContext(
        run_dir=run_dir,
        model_manifest_path=model_manifest_path,
        model_manifest=model_manifest,
        model_count=model_count,
        status_count=status_count,
        status_paths=status_paths,
        status_files_exist=status_files_exist,
        perf_report_count=perf_report_count,
        ir_count=ir_count,
        dashboard_path=dashboard_path,
        packet_path=packet_path,
        report_path=report_path,
        dashboard_text=dashboard_text,
        superset_export_dir=superset_export_dir,
        superset_export_paths=superset_export_paths,
        statuses=statuses,
    )


def coverage_model_list(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        0,
        "passed" if ctx.model_count > 0 else "missing",
        f"{ctx.model_count} models recorded in model-manifest.json",
        existing_path_strings([ctx.model_manifest_path]),
    )


def coverage_model_identities(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        1,
        (
            "passed"
            if ctx.model_count > 0
            and model_manifest_has_run_identities(ctx.model_manifest)
            else "missing"
        ),
        "each discovered model carries a run identity and source path",
        existing_path_strings([ctx.model_manifest_path]),
    )


def coverage_profile_execution(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        2,
        (
            "passed"
            if ctx.status_count == ctx.model_count and ctx.model_count > 0
            else "partial"
        ),
        f"{ctx.status_count} status.json files for {ctx.model_count} discovered models",
        existing_path_strings(ctx.status_paths),
    )


def coverage_ir_artifacts(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        3,
        "passed" if ctx.ir_count > 0 else "partial",
        f"{ctx.ir_count} models with copied IR artifacts",
        status_artifact_paths(ctx.statuses, "ir_dir"),
    )


def coverage_perf_report(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        4,
        "passed" if ctx.perf_report_count > 0 else "partial",
        f"{ctx.perf_report_count} models with tt-perf-report output",
        status_artifact_paths(ctx.statuses, "tt_perf_report"),
    )


def coverage_status_artifacts(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        5,
        (
            "passed"
            if ctx.status_count == ctx.model_count
            and ctx.model_count > 0
            and ctx.status_files_exist
            else "partial"
        ),
        "per-model status.json files contain terminal state, taxonomy, commands, artifacts, and next action",
        existing_path_strings(ctx.status_paths),
    )


def coverage_dashboard_rankings(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        6,
        "passed" if dashboard_contains_rankings(ctx.dashboard_text) else "missing",
        "dashboard.html exists and exposes ranked slow-op/model/op-type tables",
        existing_path_strings([ctx.dashboard_path]),
    )


def coverage_report_artifacts(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        7,
        (
            "passed"
            if report_artifacts_have_requirement_coverage(
                ctx.packet_path, ctx.report_path
            )
            else "missing"
        ),
        "claude-report-packet.html and report.html were written",
        existing_path_strings([ctx.packet_path, ctx.report_path]),
    )


def coverage_dashboard_search(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        8,
        (
            "passed"
            if dashboard_contains_search_controls(ctx.dashboard_text)
            else "missing"
        ),
        "dashboard.html includes search and filter controls",
        existing_path_strings([ctx.dashboard_path]),
    )


def coverage_superset_exports(ctx: RequirementCoverageContext) -> dict[str, Any]:
    return coverage_entry(
        9,
        (
            "passed"
            if ctx.superset_export_paths and len(ctx.superset_export_paths) > 0
            else "partial"
        ),
        f"{len(ctx.superset_export_paths)} collector-compatible slow-op perf reports",
        existing_path_strings(ctx.superset_export_paths),
    )


def requirement_coverage(
    run_dir: Path,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    statuses: list[dict[str, Any]],
    slow_ops: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ctx = requirement_coverage_context(run_dir, statuses)
    return [
        coverage_model_list(ctx),
        coverage_model_identities(ctx),
        coverage_profile_execution(ctx),
        coverage_ir_artifacts(ctx),
        coverage_perf_report(ctx),
        coverage_status_artifacts(ctx),
        coverage_dashboard_rankings(ctx),
        coverage_report_artifacts(ctx),
        coverage_dashboard_search(ctx),
        coverage_superset_exports(ctx),
    ]


def write_requirements_json(
    run_dir: Path,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    statuses: list[dict[str, Any]],
    slow_ops: list[dict[str, Any]],
) -> Path:
    coverage = requirement_coverage(run_dir, manifest, environment, statuses, slow_ops)
    payload = {
        "issue": {"number": ISSUE_NUMBER, "url": ISSUE_URL},
        "prd_id": PRD_ID,
        "run_id": run_dir.name,
        "generated_at": now_iso(),
        "summary": {
            "passed": sum(1 for item in coverage if item["status"] == "passed"),
            "partial": sum(1 for item in coverage if item["status"] == "partial"),
            "missing": sum(1 for item in coverage if item["status"] == "missing"),
            "total": len(coverage),
        },
        "requirements": coverage,
    }
    requirements_path = run_dir / "requirements.json"
    write_json(requirements_path, payload)
    return requirements_path


def summarize_run(
    run_dir: Path, statuses: list[dict[str, Any]], slow_ops: list[dict[str, Any]]
) -> dict[str, Any]:
    total = len(statuses)
    passed = sum(1 for row in statuses if row.get("terminal_state") == "passed")
    blocked = sum(1 for row in statuses if row.get("terminal_state") == "blocked")
    failed = sum(1 for row in statuses if row.get("terminal_state") == "failed")
    skipped = sum(1 for row in statuses if row.get("terminal_state") == "skipped")
    taxonomy_counts: dict[str, int] = defaultdict(int)
    for row in statuses:
        taxonomy_counts[row.get("taxonomy", "unknown")] += 1
    return {
        "run_dir": str(run_dir),
        "models": total,
        "passed": passed,
        "blocked": blocked,
        "failed": failed,
        "skipped": skipped,
        "slow_ops": len(slow_ops),
        "taxonomy": dict(sorted(taxonomy_counts.items())),
    }


def write_artifacts(
    run_dir: Path,
    environment: dict[str, Any],
    discovery_result: CommandResult,
    entries: list[DiscoveryEntry],
    statuses: list[dict[str, Any]],
    slow_ops: list[dict[str, Any]],
    perf_report_job_id: str = DEFAULT_COLLECTOR_JOB_ID,
) -> tuple[Path, Path, Path]:
    manifest = load_json(run_dir / "manifest.json")
    dashboard_path = run_dir / "dashboard.html"
    packet_path = run_dir / "claude-report-packet.html"
    report_path = run_dir / "report.html"
    write_json(run_dir / "environment.json", environment)
    write_json(
        run_dir / "model-manifest.json",
        {"models": [asdict(entry) for entry in entries]},
    )
    write_json(
        run_dir / "manifest.json",
        {
            **manifest,
            "summary": summarize_run(run_dir, statuses, slow_ops),
            "run": {
                **manifest.get("run", {}),
                "completed_at": now_iso(),
                "status": "completed",
            },
        },
    )
    dashboard_path.write_text(
        render_dashboard_html(run_dir, statuses, slow_ops), encoding="utf-8"
    )
    packet_path.write_text(
        render_packet_html(run_dir, manifest, environment, statuses, slow_ops),
        encoding="utf-8",
    )
    report_path.write_text(
        render_report_html(
            run_dir,
            manifest,
            environment,
            statuses,
            slow_ops,
            dashboard_path,
            packet_path,
        ),
        encoding="utf-8",
    )
    write_superset_perf_reports(run_dir, environment, slow_ops, perf_report_job_id)
    write_requirements_json(run_dir, manifest, environment, statuses, slow_ops)
    return dashboard_path, packet_path, report_path


def initialize_run_dir(output_root: Path, run_id: Optional[str]) -> Path:
    actual_run_id = run_id or make_run_id()
    run_dir = output_root / actual_run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "profiles")
    return run_dir


def maybe_find_binary(name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    found = shutil.which(name)
    return found or name


def parse_ird_timeout_seconds(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        pass
    parts = text.split("-")
    days = 0
    clock = parts[-1]
    if len(parts) == 2:
        try:
            days = int(parts[0])
        except ValueError:
            return 0
    fields = clock.split(":")
    try:
        numbers = [int(field) for field in fields]
    except ValueError:
        return 0
    if len(numbers) == 2:
        hours, minutes = numbers
        seconds = 0
    elif len(numbers) == 3:
        hours, minutes, seconds = numbers
    else:
        return 0
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def effective_remote_run_budget_seconds(args: argparse.Namespace) -> int:
    if args.run_budget_seconds > 0:
        return args.run_budget_seconds
    ird_timeout_seconds = parse_ird_timeout_seconds(args.ird_timeout)
    if ird_timeout_seconds <= DEFAULT_IRD_RUN_BUDGET_BUFFER_SECONDS:
        return 0
    return ird_timeout_seconds - DEFAULT_IRD_RUN_BUDGET_BUFFER_SECONDS


def ird_option_args(args: argparse.Namespace) -> list[str]:
    options: list[str] = [args.ird_arch]
    if args.ird_docker_image:
        options.extend(["--docker-image", args.ird_docker_image])
    if args.ird_timeout:
        options.extend(["--timeout", args.ird_timeout])
    if args.ird_cluster:
        options.extend(["--cluster", args.ird_cluster])
    if args.ird_team:
        options.extend(["--team", args.ird_team])
    if args.ird_machine:
        options.extend(["--machine", args.ird_machine])
    if args.ird_num_pcie_chips:
        options.extend(["--num-pcie-chips", str(args.ird_num_pcie_chips)])
    for extra in args.ird_extra_arg or []:
        options.extend(shlex.split(extra))
    return options


def append_optional_remote_pipeline_args(
    remote_args: list[str], args: argparse.Namespace
) -> None:
    if args.input_sequence_length > 0:
        remote_args.extend(["--input-sequence-length", str(args.input_sequence_length)])
    if args.pytest_bin:
        remote_args.extend(["--pytest-bin", args.pytest_bin])
    if args.tracy_bin:
        remote_args.extend(["--tracy-bin", args.tracy_bin])
    if args.tt_perf_report_bin:
        remote_args.extend(["--tt-perf-report-bin", args.tt_perf_report_bin])
    if args.nvidia_cohort_json:
        remote_args.extend(["--nvidia-cohort-json", args.nvidia_cohort_json])
    for benchmark_file in args.benchmark_file:
        remote_args.extend(["--benchmark-file", benchmark_file])
    for nodeid_filter in args.nodeid_filter:
        remote_args.extend(["--nodeid-filter", nodeid_filter])
    if args.max_models:
        remote_args.extend(["--max-models", str(args.max_models)])


def build_remote_pipeline_command(args: argparse.Namespace, run_id: str) -> str:
    remote_args = [
        args.ird_remote_python,
        "tests/benchmark/scripts/ttxla_profile_pipeline.py",
        "--repo-root",
        args.ird_remote_repo_root,
        "--output-root",
        args.ird_remote_output_root,
        "--run-id",
        run_id,
        "--target",
        "local",
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--discovery-timeout-seconds",
        str(args.discovery_timeout_seconds),
        "--readiness-timeout-seconds",
        str(args.readiness_timeout_seconds),
        "--batch-size",
        str(args.batch_size),
        "--num-layers",
        str(args.num_layers),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--max-raw-artifact-bytes",
        str(args.max_raw_artifact_bytes),
        "--perf-report-job-id",
        args.perf_report_job_id,
    ]
    remote_budget = effective_remote_run_budget_seconds(args)
    if remote_budget > 0:
        remote_args.extend(["--run-budget-seconds", str(remote_budget)])
    append_optional_remote_pipeline_args(remote_args, args)
    remote_args.append("run")
    setup = args.ird_remote_setup.strip()
    command = (
        f"cd {shlex.quote(args.ird_remote_repo_root)} && {shell_join(remote_args)}"
    )
    if setup:
        command = f"{setup} && {command}"
    return command


def build_ird_run_command(args: argparse.Namespace, remote_command: str) -> list[str]:
    return [args.ird_bin, "run", *ird_option_args(args), remote_command]


def build_ird_reserve_command(args: argparse.Namespace) -> list[str]:
    return [args.ird_bin, "reserve", *ird_option_args(args)]


def first_payload_value(payload: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def parse_ird_reservation_json(stdout: str, stderr: str) -> IrdReservation:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict):
        reservation_id = first_payload_value(
            payload, ("reservation_id", "reservationId", "id")
        )
        target_host = first_payload_value(
            payload, ("target_host", "targetHost", "host", "hostname")
        )
        if reservation_id or target_host:
            return IrdReservation(reservation_id, target_host, stdout, stderr)
    return IrdReservation("", "", stdout, stderr)


def parse_ird_reservation_text(stdout: str, stderr: str) -> IrdReservation:
    text = "\n".join(part for part in (stdout, stderr) if part)
    id_match = re.search(
        r"(?:reservation(?:[_ -]?id)?|id)\s*[:=]\s*([A-Za-z0-9._:-]+)",
        text,
        re.IGNORECASE,
    )
    host_match = re.search(
        r"(?:host|hostname|target)\s*[:=]\s*([A-Za-z0-9._:-]+)", text, re.IGNORECASE
    )
    return IrdReservation(
        reservation_id=id_match.group(1) if id_match else "",
        target_host=host_match.group(1) if host_match else "",
        raw_stdout=stdout,
        raw_stderr=stderr,
    )


def parse_ird_reservation(stdout: str, stderr: str = "") -> IrdReservation:
    parsed = parse_ird_reservation_json(stdout, stderr)
    if parsed.reservation_id or parsed.target_host:
        return parsed
    return parse_ird_reservation_text(stdout, stderr)


def run_template_command(
    template: str,
    variables: dict[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    stage: str,
    command_trace_path: Path,
    timeout_seconds: int,
) -> CommandResult:
    command = split_template_command(template, variables)
    return run_subprocess(
        command=command,
        cwd=cwd,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stage=stage,
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
    )


def infer_nested_ird_returncode(run_dir: Path) -> tuple[Optional[int], str]:
    manifest = load_json(run_dir / "manifest.json")
    if not manifest:
        return None, "nested manifest was not available"

    run = manifest.get("run") or {}
    summary = manifest.get("summary") or {}
    if not run.get("completed_at"):
        return None, "nested manifest did not record completed_at"

    if run.get("status") == "environment_blocked":
        return 2, "nested manifest terminalized with readiness blocker"
    if summary.get("discovery_failed"):
        return 2, "nested manifest terminalized with discovery failure"
    readiness = summary.get("readiness")
    if isinstance(readiness, dict) and readiness.get("ok") is False:
        return 2, "nested manifest terminalized with readiness failure"

    return 0, "nested manifest indicates the remote pipeline terminalized"


def parse_ird_job_id(stdout: str, stderr: str) -> str:
    text = "\n".join(part for part in (stdout, stderr) if part)
    patterns = [
        r"Job submitted successfully with ID:\s*([0-9]+)",
        r"Submitted batch job\s+([0-9]+)",
        r"\bJOBID\s*[:=]\s*([0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def initialize_ird_run(
    args: argparse.Namespace,
) -> tuple[Path, str, Path, Path, Path, str, dict[str, str], dict[str, Any]]:
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root()
    run_id = args.run_id.strip() if args.run_id else make_run_id()
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else Path(args.output_root) / run_id
    )
    ensure_dir(run_dir)
    ird_dir = ensure_dir(run_dir / "ird")
    command_trace_path = run_dir / "command-trace.jsonl"
    remote_command = build_remote_pipeline_command(args, run_id)
    variables = {
        "run_id": run_id,
        "repo_root": str(root),
        "remote_repo_root": args.ird_remote_repo_root,
        "remote_output_root": args.ird_remote_output_root,
        "remote_command": remote_command,
        "reservation_id": "",
        "target_host": "",
        "tag": args.ird_tag or run_id,
    }
    lifecycle: dict[str, Any] = {
        "target": "ird",
        "mode": args.ird_mode,
        "run_id": run_id,
        "created_at": now_iso(),
        "remote_command": remote_command,
        "pre_cleanup": [],
        "post_cleanup": [],
        "reservation": {},
        "remote_run": {},
        "release": {},
        "timeout_cleanup": {},
    }
    return (
        root,
        run_id,
        run_dir,
        ird_dir,
        command_trace_path,
        remote_command,
        variables,
        lifecycle,
    )


def run_ird_cleanup_templates(
    templates: Iterable[str],
    variables: dict[str, str],
    root: Path,
    ird_dir: Path,
    command_trace_path: Path,
    timeout_seconds: int,
    stage_prefix: str,
) -> list[dict[str, Any]]:
    results = []
    for index, template in enumerate(templates, start=1):
        result = run_template_command(
            template,
            variables,
            root,
            ird_dir / f"{stage_prefix}-{index}.out",
            ird_dir / f"{stage_prefix}-{index}.err",
            f"ird-{stage_prefix}-{index}",
            command_trace_path,
            timeout_seconds,
        )
        results.append(asdict(result))
    return results


def cleanup_timed_out_ird_run(
    result: CommandResult,
    run_dir: Path,
    root: Path,
    ird_dir: Path,
    command_trace_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    nested_returncode, nested_note = infer_nested_ird_returncode(run_dir)
    if nested_returncode is not None:
        result.returncode = nested_returncode
        result.note = f"{result.note}; {nested_note}"
        return {}

    job_id = parse_ird_job_id(
        safe_read_text(ird_dir / "ird-run.out"),
        safe_read_text(ird_dir / "ird-run.err"),
    )
    if not job_id:
        result.note = (
            f"{result.note}; nested manifest was not terminalized; "
            "could not find IRD job id for scheduler cleanup"
        )
        return {}

    timeout_cleanup = run_subprocess(
        command=["scancel", job_id],
        cwd=root,
        stdout_path=ird_dir / "ird-timeout-cleanup.out",
        stderr_path=ird_dir / "ird-timeout-cleanup.err",
        stage="ird-timeout-cleanup",
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
    )
    result.note = (
        f"{result.note}; nested manifest was not terminalized; "
        f"requested scancel for IRD job {job_id}"
    )
    return {
        **asdict(timeout_cleanup),
        "command": shell_join(timeout_cleanup.command),
        "job_id": job_id,
    }


def execute_ird_run_mode(
    args: argparse.Namespace,
    root: Path,
    run_dir: Path,
    ird_dir: Path,
    remote_command: str,
    command_trace_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = build_ird_run_command(args, remote_command)
    result = run_subprocess(
        command=command,
        cwd=root,
        stdout_path=ird_dir / "ird-run.out",
        stderr_path=ird_dir / "ird-run.err",
        stage="ird-run",
        command_trace_path=command_trace_path,
        timeout_seconds=args.ird_job_timeout_seconds,
    )
    timeout_cleanup = {}
    if result.timed_out:
        timeout_cleanup = cleanup_timed_out_ird_run(
            result,
            run_dir,
            root,
            ird_dir,
            command_trace_path,
            args.ird_control_timeout_seconds,
        )
    remote_run = asdict(result)
    remote_run["command"] = shell_join(result.command)
    return remote_run, timeout_cleanup


def execute_ird_reserved_mode(
    args: argparse.Namespace,
    variables: dict[str, str],
    root: Path,
    ird_dir: Path,
    command_trace_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reserve_result = run_subprocess(
        command=build_ird_reserve_command(args),
        cwd=root,
        stdout_path=ird_dir / "ird-reserve.out",
        stderr_path=ird_dir / "ird-reserve.err",
        stage="ird-reserve",
        command_trace_path=command_trace_path,
        timeout_seconds=args.ird_control_timeout_seconds,
    )
    reservation = parse_ird_reservation(
        safe_read_text(ird_dir / "ird-reserve.out"),
        safe_read_text(ird_dir / "ird-reserve.err"),
    )
    variables["reservation_id"] = reservation.reservation_id
    variables["target_host"] = reservation.target_host
    reservation_record = {
        **asdict(reserve_result),
        "command": shell_join(reserve_result.command),
        "reservation_id": reservation.reservation_id,
        "target_host": reservation.target_host,
    }
    if not reserve_result.ok or not reservation.reservation_id:
        return (
            reservation_record,
            {
                "stage": "ird-reserved-run",
                "returncode": 2,
                "note": "reservation failed or did not expose a reservation id",
            },
            {},
        )

    run_template = args.ird_reserved_run_command or (
        "{ird_bin} run --reservation-id {reservation_id} -- {remote_command}"
    )
    variables["ird_bin"] = args.ird_bin
    run_result = run_template_command(
        run_template,
        variables,
        root,
        ird_dir / "ird-reserved-run.out",
        ird_dir / "ird-reserved-run.err",
        "ird-reserved-run",
        command_trace_path,
        args.ird_job_timeout_seconds,
    )
    release_template = args.ird_release_command or "{ird_bin} release {reservation_id}"
    release_result = run_template_command(
        release_template,
        variables,
        root,
        ird_dir / "ird-release.out",
        ird_dir / "ird-release.err",
        "ird-release",
        command_trace_path,
        args.ird_control_timeout_seconds,
    )
    return reservation_record, asdict(run_result), asdict(release_result)


def capture_ird_environment(
    args: argparse.Namespace, root: Path, ird_dir: Path
) -> dict[str, Any]:
    environment = capture_environment(
        root,
        maybe_find_binary("tracy", args.tracy_bin),
        maybe_find_binary("tt-perf-report", args.tt_perf_report_bin),
        args.pytest_bin or "pytest",
    )
    environment["target"] = {
        "scope": "ird",
        "mode": args.ird_mode,
        "arch": args.ird_arch,
        "cluster": args.ird_cluster,
        "team": args.ird_team,
        "machine": args.ird_machine,
        "num_pcie_chips": args.ird_num_pcie_chips,
        "docker_image": args.ird_docker_image,
        "lifecycle": str(ird_dir / "ird-lifecycle.json"),
    }
    return environment


def finalize_partial_ird_run(
    run_dir: Path, root: Path, environment: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    remote_manifest = load_json(run_dir / "manifest.json")
    remote_run = remote_manifest.get("run") or {}
    remote_summary = remote_manifest.get("summary") or {}
    model_manifest = load_json(run_dir / "model-manifest.json")
    model_count = len(model_manifest.get("models", []) or [])
    status_count = len(load_model_statuses(run_dir))
    needs_finalization = model_count > 0 and (
        not remote_run.get("completed_at")
        or not remote_summary
        or status_count < model_count
        or not (run_dir / "dashboard.html").exists()
    )
    if not needs_finalization:
        return False, remote_manifest

    finalized = finalize_partial_run_from_manifest(
        run_dir=run_dir,
        repo=root,
        environment=environment,
        reason=(
            "IRD wrapper finalized the run after the nested pipeline did not "
            "complete all per-model statuses or report artifacts"
        ),
    )
    return finalized, load_json(run_dir / "manifest.json")


def execute_ird_pipeline(args: argparse.Namespace) -> int:
    (
        root,
        run_id,
        run_dir,
        ird_dir,
        command_trace_path,
        remote_command,
        variables,
        lifecycle,
    ) = initialize_ird_run(args)
    lifecycle["pre_cleanup"] = run_ird_cleanup_templates(
        args.ird_pre_cleanup_command or [],
        variables,
        root,
        ird_dir,
        command_trace_path,
        args.ird_control_timeout_seconds,
        "pre-cleanup",
    )

    if args.ird_mode == "run":
        remote_run, timeout_cleanup = execute_ird_run_mode(
            args, root, run_dir, ird_dir, remote_command, command_trace_path
        )
        lifecycle["remote_run"] = remote_run
        lifecycle["timeout_cleanup"] = timeout_cleanup
    else:
        reservation, remote_run, release = execute_ird_reserved_mode(
            args, variables, root, ird_dir, command_trace_path
        )
        lifecycle["reservation"] = reservation
        lifecycle["remote_run"] = remote_run
        lifecycle["release"] = release

    lifecycle["post_cleanup"] = run_ird_cleanup_templates(
        args.ird_post_cleanup_command or [],
        variables,
        root,
        ird_dir,
        command_trace_path,
        args.ird_control_timeout_seconds,
        "post-cleanup",
    )
    lifecycle["completed_at"] = now_iso()
    write_json(ird_dir / "ird-lifecycle.json", lifecycle)
    environment = capture_ird_environment(args, root, ird_dir)
    write_json(run_dir / "environment.json", environment)
    finalized_partial_run, remote_manifest = finalize_partial_ird_run(
        run_dir, root, environment
    )
    ird_summary = {
        "target": "ird",
        "mode": args.ird_mode,
        "remote_run_returncode": lifecycle.get("remote_run", {}).get("returncode"),
        "lifecycle": str(ird_dir / "ird-lifecycle.json"),
        "manual_cleanup": args.ird_release_command or "ird release <reservation_id>",
        "finalized_partial_run": finalized_partial_run,
    }
    if remote_manifest.get("summary"):
        ird_summary["remote_summary"] = remote_manifest["summary"]
    update_manifest_summary(
        run_dir,
        ird_summary,
    )
    return 0 if lifecycle.get("remote_run", {}).get("returncode") == 0 else 2


def initialize_local_pipeline(args: argparse.Namespace) -> LocalPipelineContext:
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root()
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else initialize_run_dir(Path(args.output_root), args.run_id)
    )
    ensure_dir(run_dir)
    ir_dump_root = ensure_dir(run_dir / "collected_irs")
    pytest_command = args.pytest_bin or "pytest"
    tracy_bin = maybe_find_binary("tracy", args.tracy_bin)
    tt_perf_report_bin = maybe_find_binary("tt-perf-report", args.tt_perf_report_bin)
    run_deadline = (
        time.monotonic() + args.run_budget_seconds
        if args.run_budget_seconds > 0
        else None
    )
    environment = capture_environment(
        root, tracy_bin, tt_perf_report_bin, pytest_command
    )
    environment["ir_dump_root"] = str(ir_dump_root)
    return LocalPipelineContext(
        root=root,
        run_dir=run_dir,
        ir_dump_root=ir_dump_root,
        python_bin=sys.executable,
        pytest_command=pytest_command,
        tracy_bin=tracy_bin,
        tt_perf_report_bin=tt_perf_report_bin,
        command_trace_path=run_dir / "command-trace.jsonl",
        run_deadline=run_deadline,
        environment=environment,
    )


def run_local_readiness(
    args: argparse.Namespace, context: LocalPipelineContext
) -> list[CommandResult]:
    readiness_results = run_tool_readiness_checks(
        repo=context.root,
        run_dir=context.run_dir,
        command_trace_path=context.command_trace_path,
        python_bin=context.python_bin,
        pytest_command=context.pytest_command,
        tracy_bin=context.tracy_bin,
        tt_perf_report_bin=context.tt_perf_report_bin,
        timeout_seconds=args.readiness_timeout_seconds,
    )
    context.environment["readiness"] = readiness_summary(readiness_results)
    return readiness_results


def discover_selected_entries(
    args: argparse.Namespace, context: LocalPipelineContext
) -> tuple[list[DiscoveryEntry], CommandResult]:
    if args.nvidia_cohort_json:
        cohort_path = Path(args.nvidia_cohort_json)
        if not cohort_path.is_absolute():
            cohort_path = context.root / cohort_path
        entries, discovery_result = select_nvidia_cohort_entries(
            repo=context.root,
            run_dir=context.run_dir,
            cohort_path=cohort_path,
            run_id=context.run_dir.name,
            python_bin=context.python_bin,
            command_trace_path=context.command_trace_path,
            timeout_seconds=args.discovery_timeout_seconds,
            validate_collection=not args.nvidia_skip_collection_validation,
            source_branch_checkout=args.nvidia_source_branch_checkout,
        )
        return (
            select_discovery_entries(entries, args.nodeid_filter, args.max_models),
            discovery_result,
        )
    entries, discovery_result = discover_models(
        repo=context.root,
        run_id=context.run_dir.name,
        python_bin=context.python_bin,
        command_trace_path=context.command_trace_path,
        benchmark_paths=selected_benchmark_files(context.root, args.benchmark_file),
        timeout_seconds=args.discovery_timeout_seconds,
    )
    return (
        select_discovery_entries(entries, args.nodeid_filter, args.max_models),
        discovery_result,
    )


def discovery_failed_without_entries(
    run_dir: Path, discovery_result: CommandResult, entries: list[DiscoveryEntry]
) -> bool:
    if discovery_result.returncode in (0, None) or entries:
        return False
    update_manifest_summary(
        run_dir,
        {
            "discovery_failed": True,
            "returncode": discovery_result.returncode,
            "note": discovery_result.note,
            "models": 0,
            "slow_ops": 0,
        },
    )
    return True


def remaining_model_timeout(
    default_timeout_seconds: int, run_deadline: Optional[float]
) -> Optional[int]:
    if run_deadline is None:
        return default_timeout_seconds
    remaining_seconds = int(run_deadline - time.monotonic())
    if remaining_seconds <= 0:
        return None
    return min(default_timeout_seconds, remaining_seconds)


def benchmark_kwargs_from_args(args: argparse.Namespace) -> dict[str, int]:
    return {
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "max_output_tokens": args.max_output_tokens,
        "input_sequence_length": args.input_sequence_length,
    }


def profile_selected_entries(
    args: argparse.Namespace,
    context: LocalPipelineContext,
    entries: list[DiscoveryEntry],
) -> None:
    benchmark_kwargs = benchmark_kwargs_from_args(args)
    for entry in entries:
        model_timeout_seconds = remaining_model_timeout(
            args.timeout_seconds, context.run_deadline
        )
        if model_timeout_seconds is None:
            write_unprofiled_model_status(
                entry=entry,
                repo=context.root,
                run_dir=context.run_dir,
                taxonomy=TAXONOMY_NOT_RUN,
                reason=(
                    "run budget was exhausted before profiling started for this model"
                ),
            )
            continue
        if args.nvidia_source_branch_checkout and entry.source_branch:
            checkout_results = checkout_nvidia_source_branch(
                repo=context.root,
                source_branch=entry.source_branch,
                run_id=entry.run_identity,
                command_trace_path=context.command_trace_path,
                timeout_seconds=args.discovery_timeout_seconds,
            )
            if any(not result.ok for result in checkout_results):
                write_unprofiled_model_status(
                    entry=entry,
                    repo=context.root,
                    run_dir=context.run_dir,
                    taxonomy=TAXONOMY_PIPELINE_ERROR,
                    reason=(
                        "tt_forge_models source_branch checkout failed before "
                        "profiling started"
                    ),
                )
                continue
        profile_one_model(
            entry=entry,
            repo=context.root,
            run_dir=context.run_dir,
            pytest_command=context.pytest_command,
            tracy_bin=context.tracy_bin,
            tt_perf_report_bin=context.tt_perf_report_bin,
            command_trace_path=context.command_trace_path,
            timeout_seconds=model_timeout_seconds,
            benchmark_kwargs=benchmark_kwargs,
            max_raw_artifact_bytes=args.max_raw_artifact_bytes,
            ir_dump_root=context.ir_dump_root,
        )


def write_final_local_artifacts(
    context: LocalPipelineContext,
    discovery_result: CommandResult,
    entries: list[DiscoveryEntry],
    perf_report_job_id: str,
) -> tuple[Path, Path, Path]:
    terminalize_missing_model_statuses(
        run_dir=context.run_dir,
        entries=entries,
        repo=context.root,
        reason="pipeline finalized before this discovered model emitted status.json",
    )
    statuses = load_model_statuses(context.run_dir)
    slow_ops = load_slow_ops(context.run_dir)
    artifact_paths = write_artifacts(
        run_dir=context.run_dir,
        environment=context.environment,
        discovery_result=discovery_result,
        entries=entries,
        statuses=statuses,
        slow_ops=slow_ops,
        perf_report_job_id=perf_report_job_id,
    )
    update_manifest_summary(
        context.run_dir, summarize_run(context.run_dir, statuses, slow_ops)
    )
    return artifact_paths


def execute_pipeline(args: argparse.Namespace) -> int:
    if args.target == "ird":
        return execute_ird_pipeline(args)

    context = initialize_local_pipeline(args)
    readiness_results = run_local_readiness(args, context)
    if any(not result.ok for result in readiness_results):
        write_readiness_blocker_artifacts(
            context.run_dir, context.environment, readiness_results
        )
        return 2

    entries, discovery_result = discover_selected_entries(args, context)
    discover_artifacts(context.run_dir, entries, discovery_result, context.environment)
    if discovery_failed_without_entries(context.run_dir, discovery_result, entries):
        return 2

    profile_selected_entries(args, context, entries)
    dashboard_path, packet_path, report_path = write_final_local_artifacts(
        context, discovery_result, entries, args.perf_report_job_id
    )
    print(f"dashboard: {dashboard_path}")
    print(f"packet: {packet_path}")
    print(f"report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TT-XLA profiling pipeline for issue #5009"
    )
    parser.add_argument(
        "--repo-root", default="", help="Path to the tt-xla repository root."
    )
    parser.add_argument(
        "--run-dir", default="", help="Existing run directory to reuse."
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder for new profiling runs.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier; defaults to a timestamped value.",
    )
    parser.add_argument(
        "--pytest-bin",
        default="",
        help="Pytest command passed to tracy -m and recorded in the environment snapshot.",
    )
    parser.add_argument(
        "--tracy-bin", default="", help="Path to tracy; defaults to PATH lookup."
    )
    parser.add_argument(
        "--tt-perf-report-bin",
        default="",
        help="Path to tt-perf-report; defaults to PATH lookup.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-model profiling timeout.",
    )
    parser.add_argument(
        "--run-budget-seconds",
        type=int,
        default=0,
        help="Total wall-clock budget for local profiling after readiness/discovery; 0 disables the run-level budget.",
    )
    parser.add_argument(
        "--discovery-timeout-seconds", type=int, default=900, help="Discovery timeout."
    )
    parser.add_argument(
        "--readiness-timeout-seconds",
        type=int,
        default=30,
        help="Timeout for pytest, Tracy, and tt-perf-report readiness checks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Bounded batch size for benchmark fixtures.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Bounded num-layers fixture value.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Bounded max-output-tokens fixture value.",
    )
    parser.add_argument(
        "--input-sequence-length",
        type=int,
        default=DEFAULT_INPUT_SEQUENCE_LENGTH,
        help="Bounded input-sequence-length fixture value for benchmarks that support it; 0 preserves benchmark defaults.",
    )
    parser.add_argument(
        "--max-raw-artifact-bytes",
        type=int,
        default=DEFAULT_MAX_RAW_ARTIFACT_BYTES,
        help="Prune raw Tracy artifacts larger than this many bytes after each profile; 0 disables pruning.",
    )
    parser.add_argument(
        "--perf-report-job-id",
        default=os.environ.get("TTXLA_PERF_REPORT_JOB_ID", DEFAULT_COLLECTOR_JOB_ID),
        help="Numeric GitHub check-run id used as the trailing id in Superset collector perf_report JSON filenames.",
    )
    parser.add_argument(
        "--benchmark-file",
        action="append",
        default=[],
        help="Benchmark file to collect; may be repeated. Defaults to the full benchmark cohort.",
    )
    parser.add_argument(
        "--nvidia-cohort-json",
        default="",
        help=(
            "Path to a NVIDIA/SILICON_PASS cohort JSON with test_case_id rows. "
            "When set, profiles matching tests/runner/test_models.py TT runner "
            "node IDs instead of collecting tests/benchmark files."
        ),
    )
    parser.add_argument(
        "--nvidia-skip-collection-validation",
        action="store_true",
        help=(
            "Use synthetic NVIDIA test_case_id to TT runner node mapping without "
            "validating those node IDs through pytest collection first. Intended "
            "only for dry mapping/debugging when the runner environment is not available."
        ),
    )
    parser.add_argument(
        "--nvidia-source-branch-checkout",
        action="store_true",
        help=(
            "For NVIDIA/SILICON_PASS cohort rows that include source_branch, "
            "checkout that tt_forge_models branch before collection and before "
            "profiling each row. This reproduces Nick's branch-pinned bringup "
            "pipeline behavior."
        ),
    )
    parser.add_argument(
        "--nodeid-filter",
        action="append",
        default=[],
        help="Only profile collected pytest node IDs containing this substring; may be repeated.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Maximum selected benchmark entries to profile; 0 means no limit.",
    )
    parser.add_argument(
        "--target",
        choices=("local", "ird"),
        default="local",
        help="Execution target for the run subcommand.",
    )
    parser.add_argument("--ird-bin", default="ird", help="IRD command path.")
    parser.add_argument(
        "--ird-mode",
        choices=("run", "reserve"),
        default="run",
        help="Use short ird run or explicit reserve/release mode.",
    )
    parser.add_argument(
        "--ird-arch", default="wormhole_b0", help="IRD architecture selector."
    )
    parser.add_argument(
        "--ird-docker-image",
        default="xla",
        help="IRD docker image or image alias.",
    )
    parser.add_argument(
        "--ird-timeout", default="45:00", help="IRD scheduler timeout value."
    )
    parser.add_argument("--ird-cluster", default="", help="Optional IRD cluster.")
    parser.add_argument("--ird-team", default="", help="Optional IRD team.")
    parser.add_argument("--ird-machine", default="", help="Optional IRD machine.")
    parser.add_argument(
        "--ird-num-pcie-chips",
        type=int,
        default=1,
        help="Number of PCIe chips requested from IRD.",
    )
    parser.add_argument(
        "--ird-extra-arg",
        action="append",
        default=[],
        help="Additional IRD option string; may be repeated.",
    )
    parser.add_argument(
        "--ird-tag",
        default="",
        help="Owner/run tag available to cleanup command templates as {tag}.",
    )
    parser.add_argument(
        "--ird-remote-repo-root",
        default="/work/tt-xla",
        help="tt-xla repo root inside the IRD container.",
    )
    parser.add_argument(
        "--ird-remote-output-root",
        default="/work/tt-xla/artifacts/prd-009/ttxla-profile",
        help="Output root inside the IRD container.",
    )
    parser.add_argument(
        "--ird-remote-python",
        default="python3",
        help="Python executable inside the IRD container.",
    )
    parser.add_argument(
        "--ird-remote-setup",
        default="source venv/activate",
        help="Shell setup command to run before the remote pipeline.",
    )
    parser.add_argument(
        "--ird-control-timeout-seconds",
        type=int,
        default=300,
        help="Timeout for IRD cleanup/reserve/release commands.",
    )
    parser.add_argument(
        "--ird-job-timeout-seconds",
        type=int,
        default=21600,
        help="Timeout for the IRD profiling job command.",
    )
    parser.add_argument(
        "--ird-pre-cleanup-command",
        action="append",
        default=[],
        help="Best-effort command template to run before IRD execution; may use {run_id}, {tag}, {reservation_id}, {target_host}.",
    )
    parser.add_argument(
        "--ird-post-cleanup-command",
        action="append",
        default=[],
        help="Best-effort command template to run after IRD execution; may use {run_id}, {tag}, {reservation_id}, {target_host}.",
    )
    parser.add_argument(
        "--ird-reserved-run-command",
        default="",
        help="Command template for reserve mode execution; defaults to 'ird run --reservation-id ...'.",
    )
    parser.add_argument(
        "--ird-release-command",
        default="",
        help="Command template for reserve mode release; defaults to 'ird release {reservation_id}'.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "discover", help="Collect benchmark models and write the run manifest."
    )
    subparsers.add_parser(
        "run",
        help="Discover models, profile them, and render dashboard/report artifacts.",
    )
    subparsers.add_parser(
        "dashboard", help="Render dashboard.html from an existing run directory."
    )
    subparsers.add_parser(
        "report-packet",
        help="Render the source packet and final report from an existing run directory.",
    )
    return parser


def execute_discover_command(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else initialize_run_dir(Path(args.output_root), args.run_id)
    )
    ensure_dir(run_dir)
    python_bin = sys.executable
    pytest_command = args.pytest_bin or "pytest"
    tracy_bin = maybe_find_binary("tracy", args.tracy_bin)
    tt_perf_report_bin = maybe_find_binary("tt-perf-report", args.tt_perf_report_bin)
    command_trace_path = run_dir / "command-trace.jsonl"
    environment = capture_environment(
        root, tracy_bin, tt_perf_report_bin, pytest_command
    )
    if args.nvidia_cohort_json:
        cohort_path = Path(args.nvidia_cohort_json)
        if not cohort_path.is_absolute():
            cohort_path = root / cohort_path
        entries, discovery_result = select_nvidia_cohort_entries(
            repo=root,
            run_dir=run_dir,
            cohort_path=cohort_path,
            run_id=run_dir.name,
            python_bin=python_bin,
            command_trace_path=command_trace_path,
            timeout_seconds=args.discovery_timeout_seconds,
            validate_collection=not args.nvidia_skip_collection_validation,
            source_branch_checkout=args.nvidia_source_branch_checkout,
        )
    else:
        entries, discovery_result = discover_models(
            repo=root,
            run_id=run_dir.name,
            python_bin=python_bin,
            command_trace_path=command_trace_path,
            timeout_seconds=args.discovery_timeout_seconds,
        )
    entries = select_discovery_entries(entries, args.nodeid_filter, args.max_models)
    discover_artifacts(run_dir, entries, discovery_result, environment)
    update_manifest_summary(
        run_dir,
        {
            "discovery_only": True,
            "returncode": discovery_result.returncode,
            "models": len(entries),
            "slow_ops": 0,
        },
    )
    print(f"manifest: {run_dir / 'manifest.json'}")
    print(f"model-manifest: {run_dir / 'model-manifest.json'}")
    if discovery_result.returncode not in (0, None) and not entries:
        return 2
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_id:
        args.run_id = args.run_id.strip()
    args.repo_root = args.repo_root or str(repo_root())
    if args.command == "run":
        return execute_pipeline(args)
    if args.command == "discover":
        return execute_discover_command(args)
    if not args.run_dir:
        parser.error("--run-dir is required for dashboard and report-packet")
    run_dir = Path(args.run_dir).resolve()
    statuses = load_model_statuses(run_dir)
    slow_ops = load_slow_ops(run_dir)
    environment = load_json(run_dir / "environment.json")
    manifest = load_json(run_dir / "manifest.json")
    if args.command == "dashboard":
        (run_dir / "dashboard.html").write_text(
            render_dashboard_html(run_dir, statuses, slow_ops), encoding="utf-8"
        )
        write_superset_perf_reports(
            run_dir, environment, slow_ops, args.perf_report_job_id
        )
        write_requirements_json(run_dir, manifest, environment, statuses, slow_ops)
        return 0
    packet_path = run_dir / "claude-report-packet.html"
    dashboard_path = run_dir / "dashboard.html"
    report_path = run_dir / "report.html"
    packet_path.write_text(
        render_packet_html(run_dir, manifest, environment, statuses, slow_ops),
        encoding="utf-8",
    )
    report_path.write_text(
        render_report_html(
            run_dir,
            manifest,
            environment,
            statuses,
            slow_ops,
            dashboard_path,
            packet_path,
        ),
        encoding="utf-8",
    )
    write_superset_perf_reports(run_dir, environment, slow_ops, args.perf_report_job_id)
    write_requirements_json(run_dir, manifest, environment, statuses, slow_ops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
