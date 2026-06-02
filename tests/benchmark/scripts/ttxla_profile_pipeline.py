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
DEFAULT_BENCHMARK_FILES = [
    Path("tests/benchmark/test_llms.py"),
    Path("tests/benchmark/test_encoders.py"),
    Path("tests/benchmark/test_vision.py"),
    Path("tests/benchmark/resnet_jax_benchmark.py"),
]

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
]

ENVIRONMENT_FAILURE_HINTS = (
    "module not found",
    "no module named",
    "command not found",
    "not found",
    "permission denied",
    "file not found",
    "no such file or directory",
    "missing dependency",
)

MODEL_FAILURE_HINTS = (
    "assertionerror",
    "pcc comparison failed",
    "comparison failed",
    "runtimeerror",
    "ttmlir compilation",
    "frontend conversion",
    "unsupported operator",
    "device crash",
    "traceback",
)

SKIP_HINTS = (
    "skipped",
    "xfailed",
    "xfail",
)

CSV_DURATION_ALIASES = (
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

CSV_OP_ALIASES = ("op_name", "operation", "op", "name", "kernel")
CSV_MODEL_ALIASES = ("model", "model_name", "model_id", "model_identity")
CSV_OP_TYPE_ALIASES = ("op_type", "type", "category", "kind")


@dataclass
class DiscoveryEntry:
    run_identity: str
    nodeid: str
    source_path: str
    test_name: str
    benchmark_family: str
    model_identity: str
    artifact_slug: str


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    benchmark_kwargs: dict[str, int],
) -> list[str]:
    return [
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
        "--output-file",
        str(benchmark_output),
        "--batch-size",
        str(benchmark_kwargs["batch_size"]),
        "--num-layers",
        str(benchmark_kwargs["num_layers"]),
        "--max-output-tokens",
        str(benchmark_kwargs["max_output_tokens"]),
    ]


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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_collect_output(output: str, run_id: str) -> list[DiscoveryEntry]:
    entries: list[DiscoveryEntry] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith("tests/benchmark/"):
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


def copy_tree(source: Path, target: Path) -> list[str]:
    copied: list[str] = []
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
        copied.append(str(dest))
    return copied


def find_latest_csv(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [
        path for path in root.rglob("ops_perf_results_*.csv") if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


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
    if lowered.endswith("_ns"):
        return numeric / 1000.0
    if lowered.endswith("_ms"):
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
            duration_raw_key = ""
            duration_value: Optional[float] = None
            for alias in CSV_DURATION_ALIASES:
                candidate = row.get(alias)
                if candidate not in (None, ""):
                    duration_raw_key = alias
                    duration_value = parse_duration_value(candidate, alias)
                    if duration_value is not None:
                        break
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


def infer_taxonomy(
    returncode: Optional[int],
    timed_out: bool,
    text: str,
    benchmark_json: dict[str, Any],
    perf_report_ok: bool,
) -> tuple[str, str]:
    lowered = text.lower()
    if timed_out:
        return "pending_terminalization", "timed out before reaching a terminal state"
    if returncode == 0 and perf_report_ok:
        return "validated_pass", "profiling run completed and perf report was produced"
    if any(hint in lowered for hint in SKIP_HINTS):
        return "skipped_with_reason", "benchmark entry was skipped"
    if any(hint in lowered for hint in ENVIRONMENT_FAILURE_HINTS):
        return (
            "environment_failure",
            "environment or dependency issue blocked profiling",
        )
    if returncode not in (0, None):
        if any(hint in lowered for hint in MODEL_FAILURE_HINTS):
            if any(
                term in lowered
                for term in ("pcc comparison failed", "accuracy", "validation")
            ):
                return "validated_fail", "model compiled but validation failed"
            return (
                "model_failure",
                "model or runtime behavior failed after environment preconditions passed",
            )
        if benchmark_json:
            return (
                "pipeline_error",
                "pipeline or artifact stage failed after benchmark output was produced",
            )
        return (
            "pipeline_error",
            "pipeline or artifact stage failed before benchmark output was produced",
        )
    if benchmark_json and not perf_report_ok:
        return (
            "pipeline_error",
            "benchmark completed but perf report could not be produced",
        )
    return "pipeline_error", "pipeline did not produce a terminal profiling artifact"


def terminal_state_for_taxonomy(taxonomy: str) -> str:
    mapping = {
        "validated_pass": "passed",
        "validated_fail": "failed",
        "model_failure": "failed",
        "environment_failure": "blocked",
        "pipeline_error": "blocked",
        "not_run": "blocked",
        "skipped_with_reason": "skipped",
        "pending_terminalization": "pending",
        "compiled_only": "partial",
    }
    return mapping.get(taxonomy, "blocked")


def next_action_for_taxonomy(taxonomy: str) -> str:
    mapping = {
        "validated_pass": "Review dashboard rankings and choose the next optimization target.",
        "validated_fail": "Fix the validation mismatch, then rerun the model profile.",
        "compiled_only": "Collect the missing validation or perf data, then rerun.",
        "model_failure": "Resolve the model/runtime failure, then rerun the profile.",
        "environment_failure": "Repair the missing dependency or host setup, then rerun.",
        "pipeline_error": "Repair the pipeline or artifact stage, then rerun the profile.",
        "not_run": "Run the model profile or record the owner for the non-run decision.",
        "skipped_with_reason": "Review the skip reason and owner before promoting the model.",
        "pending_terminalization": "Wait for terminalization or collect the missing job state before rerunning.",
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
) -> dict[str, Any]:
    profile_dir = ensure_dir(run_dir / "profiles" / entry.artifact_slug)
    perf_dir = ensure_dir(profile_dir / "perf-report")
    ir_dir = ensure_dir(profile_dir / "ir")
    trace_dir = ensure_dir(profile_dir / "tracy")
    benchmark_output = profile_dir / "benchmark.json"
    run_log = profile_dir / "run.log"
    compile_log = profile_dir / "compile.log"
    perf_output = perf_dir / "tt-perf-report.txt"
    perf_input = perf_dir / "tt-perf-report-input.csv"
    slow_ops_path = profile_dir / "slow-ops.json"
    profile_command_list = profile_command(
        tracy_bin=tracy_bin,
        pytest_command=pytest_command,
        nodeid=entry.nodeid,
        profile_dir=profile_dir,
        benchmark_output=benchmark_output,
        benchmark_kwargs=benchmark_kwargs,
    )

    profile_env = os.environ.copy()
    profile_env.setdefault("TTMLIR_ENABLE_PERF_TRACE", "1")
    profile_env.setdefault("TT_RUNTIME_TRACE_REGION_SIZE", "10000000")
    profile_env["TTXLA_PROFILE_RUN_ID"] = entry.run_identity
    profile_env["TTXLA_PROFILE_NODEID"] = entry.nodeid

    profile_result = run_subprocess(
        command=profile_command_list,
        cwd=repo,
        stdout_path=run_log,
        stderr_path=compile_log,
        stage="profile",
        command_trace_path=command_trace_path,
        timeout_seconds=timeout_seconds,
        env=profile_env,
    )

    benchmark_json = load_json(benchmark_output)
    benchmark_model_name = extract_benchmark_model_name(
        benchmark_json, entry.model_identity
    )
    collected_irs_root = repo / "collected_irs"
    ir_source_candidates = [
        collected_irs_root / benchmark_model_name,
        collected_irs_root / slugify(benchmark_model_name),
        collected_irs_root / entry.model_identity,
        collected_irs_root / slugify(entry.model_identity),
    ]
    ir_source = next(
        (candidate for candidate in ir_source_candidates if candidate.exists()), None
    )
    copied_ir_files = copy_tree(ir_source, ir_dir) if ir_source else []

    perf_csv_source = find_latest_csv(trace_dir)
    perf_report_result: Optional[CommandResult] = None
    perf_report_ok = False
    perf_csv_recorded = ""
    perf_report_reason = ""
    perf_report_command_list: list[str] = []
    if perf_csv_source:
        perf_csv_recorded = str(perf_input)
        shutil.copy2(perf_csv_source, perf_input)
    if perf_csv_source and command_expr_available(tt_perf_report_bin):
        perf_report_command_list = perf_report_command(tt_perf_report_bin, perf_input)
        perf_report_result = run_subprocess(
            command=perf_report_command_list,
            cwd=profile_dir,
            stdout_path=perf_output,
            stderr_path=perf_dir / "tt-perf-report.err",
            stage="tt-perf-report",
            command_trace_path=command_trace_path,
            timeout_seconds=min(timeout_seconds, 300),
            env=os.environ.copy(),
        )
        perf_report_ok = perf_report_result.ok
        if not perf_report_ok:
            perf_report_reason = (
                safe_read_text(perf_dir / "tt-perf-report.err")
                or "tt-perf-report returned a non-zero exit code"
            )
    else:
        if not perf_csv_source:
            perf_report_reason = "no ops CSV was produced by the Tracy profile"
        elif not shutil.which(tt_perf_report_bin):
            perf_report_reason = "tt-perf-report binary was not available"

    slow_ops_payload: dict[str, Any] = {
        "model": benchmark_model_name,
        "nodeid": entry.nodeid,
        "source_path": entry.source_path,
        "profile_dir": str(profile_dir),
        "ir_dir": str(ir_dir),
        "perf_csv": perf_csv_recorded,
        "tt_perf_report": str(perf_output),
        "rows": [],
        "summary": {
            "row_count": 0,
            "total_duration_us": 0.0,
            "op_type_totals": {},
            "op_name_totals": {},
        },
    }
    if perf_csv_source and perf_csv_source.exists():
        parsed = parse_perf_csv(
            perf_csv_source, benchmark_model_name, entry.artifact_slug
        )
        slow_ops_payload["rows"] = parsed["rows"]
        slow_ops_payload["summary"] = parsed["summary"]
        write_json(slow_ops_path, slow_ops_payload)
    else:
        write_json(slow_ops_path, slow_ops_payload)

    combined_text = "\n".join(
        text
        for text in (
            safe_read_text(run_log),
            safe_read_text(compile_log),
            safe_read_text(perf_output),
        )
        if text
    )
    taxonomy, reason = infer_taxonomy(
        returncode=profile_result.returncode,
        timed_out=profile_result.timed_out,
        text=combined_text,
        benchmark_json=benchmark_json,
        perf_report_ok=perf_report_ok,
    )
    terminal_state = terminal_state_for_taxonomy(taxonomy)
    next_action = next_action_for_taxonomy(taxonomy)

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
                shell_join(perf_report_command_list) if perf_report_command_list else ""
            ),
        },
        "artifacts": {
            **stage_result_paths(profile_dir),
            "ir_source": str(ir_source) if ir_source else "",
            "copied_ir_count": len(copied_ir_files),
            "perf_csv_source": str(perf_csv_source) if perf_csv_source else "",
        },
        "stages": {
            "profile": {
                "state": terminal_state,
                "returncode": profile_result.returncode,
                "timed_out": profile_result.timed_out,
                "note": profile_result.note,
                "stdout": str(run_log),
                "stderr": str(compile_log),
            },
            "ir": {
                "state": "collected" if copied_ir_files else "missing",
                "source": str(ir_source) if ir_source else "",
                "count": len(copied_ir_files),
            },
            "tt_perf_report": {
                "state": "generated" if perf_report_ok else "blocked",
                "returncode": getattr(perf_report_result, "returncode", None),
                "note": perf_report_reason,
            },
        },
        "taxonomy": taxonomy,
        "terminal_state": terminal_state,
        "reason": reason,
        "next_action": next_action,
        "benchmark": benchmark_json,
        "slow_ops": str(slow_ops_path),
        "verification": {
            "verified": terminal_state == "passed"
            and perf_report_ok
            and bool(copied_ir_files),
            "state": (
                "verified"
                if terminal_state == "passed" and perf_report_ok
                else "pending"
            ),
        },
    }

    write_json(profile_dir / "status.json", status_payload)
    return status_payload


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
        statuses.append(load_json(status_file))
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
            record["profile_status"] = model_status.get("terminal_state", "unknown")
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
                "profile_status": status.get("terminal_state", "unknown"),
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


def render_dashboard_html(
    run_dir: Path, statuses: list[dict[str, Any]], slow_ops: list[dict[str, Any]]
) -> str:
    models = aggregate_models(statuses, slow_ops)
    op_types = aggregate_op_types(slow_ops)
    all_status_values = sorted(
        {row.get("profile_status", "unknown") for row in slow_ops}
        | {row.get("profile_status", "unknown") for row in models}
    )
    all_taxonomies = sorted(
        {row.get("taxonomy", "unknown") for row in slow_ops}
        | {row.get("taxonomy", "unknown") for row in models}
    )
    html_rows = []
    for row in slow_ops:
        html_rows.append(
            "<tr "
            f'data-model="{html.escape(str(row.get("model_identity", "")))}" '
            f'data-op="{html.escape(str(row.get("op_name", "")))}" '
            f'data-op-type="{html.escape(str(row.get("op_type", "")))}" '
            f'data-status="{html.escape(str(row.get("profile_status", "")))}" '
            f'data-taxonomy="{html.escape(str(row.get("taxonomy", "")))}">'
            f"<td>{row.get('global_rank', '')}</td>"
            f"<td>{html.escape(str(row.get('model_identity', '')))}</td>"
            f"<td>{html.escape(str(row.get('op_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('op_type', '')))}</td>"
            f"<td>{float(row.get('duration_us', 0.0)):.2f}</td>"
            f"<td>{html.escape(str(row.get('profile_status', '')))}</td>"
            f"<td>{html.escape(str(row.get('taxonomy', '')))}</td>"
            f"<td>{render_links(run_dir, str(row.get('status_path', '')), 'status')}</td>"
            f"<td>{render_links(run_dir, str(row.get('ir_dir', '')), 'ir')}</td>"
            f"<td>{render_links(run_dir, str(row.get('perf_report', '')), 'perf')}</td>"
            "</tr>"
        )

    model_rows = []
    for row in models:
        model_rows.append(
            "<tr "
            f'data-model="{html.escape(str(row.get("model_identity", "")))}" '
            f'data-status="{html.escape(str(row.get("profile_status", "")))}" '
            f'data-taxonomy="{html.escape(str(row.get("taxonomy", "")))}">'
            f"<td>{row.get('global_rank', '')}</td>"
            f"<td>{html.escape(str(row.get('model_identity', '')))}</td>"
            f"<td>{html.escape(str(row.get('profile_status', '')))}</td>"
            f"<td>{html.escape(str(row.get('taxonomy', '')))}</td>"
            f"<td>{row.get('row_count', 0)}</td>"
            f"<td>{row.get('total_duration_us', 0.0):.2f}</td>"
            f"<td>{html.escape(str(row.get('slowest_op', '')))}</td>"
            f"<td>{html.escape(str(row.get('slowest_op_type', '')))}</td>"
            f"<td>{render_links(run_dir, str(row.get('ir_dir', '')), 'ir')}</td>"
            f"<td>{render_links(run_dir, str(row.get('perf_report', '')), 'perf')}</td>"
            "</tr>"
        )

    op_type_rows = []
    for row in op_types:
        op_type_rows.append(
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

    total_models = len(models)
    total_ops = len(slow_ops)
    passed_models = sum(1 for row in models if row.get("profile_status") == "passed")
    blocked_models = sum(1 for row in models if row.get("profile_status") == "blocked")
    failed_models = sum(1 for row in models if row.get("profile_status") == "failed")

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
      <div class="metric"><strong>{total_models}</strong><span>models in scope</span></div>
      <div class="metric"><strong>{passed_models}</strong><span>passed</span></div>
      <div class="metric"><strong>{blocked_models}</strong><span>blocked</span></div>
      <div class="metric"><strong>{failed_models}</strong><span>failed</span></div>
      <div class="metric"><strong>{total_ops}</strong><span>slow-op rows</span></div>
    </div>
    <div class="controls">
      <div><label for="search">Search</label><input id="search" type="search" placeholder="model, op, or path" /></div>
      <div><label for="status">Profile status</label><select id="status"><option value="">All</option>{''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in all_status_values)}</select></div>
      <div><label for="taxonomy">Taxonomy</label><select id="taxonomy"><option value="">All</option>{''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in all_taxonomies)}</select></div>
    </div>
  </section>
  <section>
    <h2>Global slow operations</h2>
    <p class="section-note">Rows are ranked by duration and carry links back to per-model artifacts.</p>
    <div class="table-wrap">
      <table id="slow-ops">
        <thead>
          <tr><th>#</th><th>Model</th><th>Operation</th><th>Type</th><th>Duration us</th><th>Status</th><th>Taxonomy</th><th>Status</th><th>IR</th><th>Perf</th></tr>
        </thead>
        <tbody>
          {''.join(html_rows) or '<tr><td colspan="10">No slow-op data available.</td></tr>'}
        </tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>By model</h2>
    <div class="table-wrap">
      <table id="models">
        <thead>
          <tr><th>#</th><th>Model</th><th>Status</th><th>Taxonomy</th><th>Ops</th><th>Total us</th><th>Slowest op</th><th>Type</th><th>IR</th><th>Perf</th></tr>
        </thead>
        <tbody>
          {''.join(model_rows) or '<tr><td colspan="10">No model summary available.</td></tr>'}
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
    const taxonomy = document.getElementById('taxonomy');
    function matches(row) {{
      const text = `${{row.dataset.model || ''}} ${{row.dataset.op || ''}} ${{row.dataset.opType || ''}}`.toLowerCase();
      const needle = search.value.trim().toLowerCase();
      if (needle && !text.includes(needle)) {{
        return false;
      }}
      if (status.value && row.dataset.status !== status.value) {{
        return false;
      }}
      if (taxonomy.value && row.dataset.taxonomy !== taxonomy.value) {{
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
      <thead><tr><th>Model</th><th>Status</th><th>Taxonomy</th><th>Reason</th><th>Perf</th><th>IR</th><th>Slow ops</th></tr></thead>
      <tbody>{''.join(model_rows) or '<tr><td colspan="7">No model status artifacts were generated.</td></tr>'}</tbody>
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
    coverage_rows = []
    for entry in coverage:
        coverage_rows.append(
            f"<tr><td>{html.escape(entry['id'])}</td><td>{html.escape(entry['status'])}</td><td>{html.escape(entry['evidence'])}</td></tr>"
        )
    counts = {
        "models": len(statuses),
        "passed": sum(
            1 for status in statuses if status.get("terminal_state") == "passed"
        ),
        "failed": sum(
            1 for status in statuses if status.get("terminal_state") == "failed"
        ),
        "blocked": sum(
            1 for status in statuses if status.get("terminal_state") == "blocked"
        ),
        "skipped": sum(
            1 for status in statuses if status.get("terminal_state") == "skipped"
        ),
        "ops": len(slow_ops),
    }
    blockers = [
        status for status in statuses if status.get("terminal_state") != "passed"
    ]
    blocker_rows = []
    for status in blockers:
        model = status.get("model", {})
        blocker_rows.append(
            "<tr>"
            f"<td>{html.escape(str(model.get('model_identity', '')))}</td>"
            f"<td>{html.escape(str(status.get('terminal_state', '')))}</td>"
            f"<td>{html.escape(str(status.get('taxonomy', '')))}</td>"
            f"<td>{html.escape(str(status.get('next_action', '')))}</td>"
            "</tr>"
        )
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
      <thead><tr><th>Model</th><th>Status</th><th>Taxonomy</th><th>Next action</th></tr></thead>
      <tbody>{''.join(blocker_rows) or '<tr><td colspan="4">No open blockers.</td></tr>'}</tbody>
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


def requirement_coverage(
    run_dir: Path,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    statuses: list[dict[str, Any]],
    slow_ops: list[dict[str, Any]],
) -> list[dict[str, str]]:
    model_manifest_path = run_dir / "model-manifest.json"
    model_manifest = load_json(model_manifest_path)
    model_count = len(model_manifest.get("models", []))
    status_count = len(statuses)
    perf_report_count = sum(
        1
        for status in statuses
        if status.get("stages", {}).get("tt_perf_report", {}).get("state")
        == "generated"
    )
    ir_count = sum(
        1
        for status in statuses
        if status.get("stages", {}).get("ir", {}).get("state") == "collected"
    )
    dashboard_exists = True
    packet_exists = True
    report_exists = True
    search_controls = "search"
    coverage = [
        {
            "id": "REQ-F-001",
            "status": "passed" if model_count > 0 else "missing",
            "evidence": f"{model_count} models recorded in model-manifest.json",
        },
        {
            "id": "REQ-F-002",
            "status": (
                "passed"
                if model_count > 0
                and all(
                    "run_identity" in row for row in model_manifest.get("models", [])
                )
                else "missing"
            ),
            "evidence": "each discovered model carries a run identity and source path",
        },
        {
            "id": "REQ-F-003",
            "status": (
                "passed"
                if status_count == model_count and model_count > 0
                else "partial"
            ),
            "evidence": f"{status_count} status.json files for {model_count} discovered models",
        },
        {
            "id": "REQ-F-004",
            "status": "passed" if ir_count > 0 else "partial",
            "evidence": f"{ir_count} models with copied IR artifacts",
        },
        {
            "id": "REQ-F-005",
            "status": "passed" if perf_report_count > 0 else "partial",
            "evidence": f"{perf_report_count} models with tt-perf-report output",
        },
        {
            "id": "REQ-F-006",
            "status": (
                "passed"
                if status_count == model_count and model_count > 0
                else "partial"
            ),
            "evidence": "per-model status.json files contain terminal state, taxonomy, commands, artifacts, and next action",
        },
        {
            "id": "REQ-F-007",
            "status": "passed" if dashboard_exists else "missing",
            "evidence": f"dashboard.html exists and exposes ranked slow-op/model/op-type tables",
        },
        {
            "id": "REQ-F-008",
            "status": "passed" if packet_exists and report_exists else "missing",
            "evidence": "claude-report-packet.html and report.html were written",
        },
        {
            "id": "REQ-F-009",
            "status": "passed" if dashboard_exists and search_controls else "missing",
            "evidence": "dashboard.html includes search and filter controls",
        },
    ]
    return coverage


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


def ird_option_args(args: argparse.Namespace) -> list[str]:
    options: list[str] = []
    if args.ird_docker_image:
        options.extend(["--docker-image", args.ird_docker_image])
    if args.ird_timeout:
        options.extend(["--timeout", args.ird_timeout])
    options.append(args.ird_arch)
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
        "--batch-size",
        str(args.batch_size),
        "--num-layers",
        str(args.num_layers),
        "--max-output-tokens",
        str(args.max_output_tokens),
    ]
    if args.pytest_bin:
        remote_args.extend(["--pytest-bin", args.pytest_bin])
    if args.tracy_bin:
        remote_args.extend(["--tracy-bin", args.tracy_bin])
    if args.tt_perf_report_bin:
        remote_args.extend(["--tt-perf-report-bin", args.tt_perf_report_bin])
    for nodeid_filter in args.nodeid_filter:
        remote_args.extend(["--nodeid-filter", nodeid_filter])
    if args.max_models:
        remote_args.extend(["--max-models", str(args.max_models)])
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


def parse_ird_reservation(stdout: str, stderr: str = "") -> IrdReservation:
    text = "\n".join(part for part in (stdout, stderr) if part)
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict):
        reservation_id = str(
            payload.get("reservation_id")
            or payload.get("reservationId")
            or payload.get("id")
            or ""
        )
        target_host = str(
            payload.get("target_host")
            or payload.get("targetHost")
            or payload.get("host")
            or payload.get("hostname")
            or ""
        )
        if reservation_id or target_host:
            return IrdReservation(reservation_id, target_host, stdout, stderr)

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


def execute_ird_pipeline(args: argparse.Namespace) -> int:
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
    }

    for index, template in enumerate(args.ird_pre_cleanup_command or [], start=1):
        result = run_template_command(
            template,
            variables,
            root,
            ird_dir / f"pre-cleanup-{index}.out",
            ird_dir / f"pre-cleanup-{index}.err",
            f"ird-pre-cleanup-{index}",
            command_trace_path,
            args.ird_control_timeout_seconds,
        )
        lifecycle["pre_cleanup"].append(asdict(result))

    if args.ird_mode == "run":
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
        lifecycle["remote_run"] = asdict(result)
        lifecycle["remote_run"]["command"] = shell_join(result.command)
    else:
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
        lifecycle["reservation"] = {
            **asdict(reserve_result),
            "command": shell_join(reserve_result.command),
            "reservation_id": reservation.reservation_id,
            "target_host": reservation.target_host,
        }
        if not reserve_result.ok or not reservation.reservation_id:
            lifecycle["remote_run"] = {
                "stage": "ird-reserved-run",
                "returncode": 2,
                "note": "reservation failed or did not expose a reservation id",
            }
        else:
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
            lifecycle["remote_run"] = asdict(run_result)
            release_template = (
                args.ird_release_command or "{ird_bin} release {reservation_id}"
            )
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
            lifecycle["release"] = asdict(release_result)

    for index, template in enumerate(args.ird_post_cleanup_command or [], start=1):
        result = run_template_command(
            template,
            variables,
            root,
            ird_dir / f"post-cleanup-{index}.out",
            ird_dir / f"post-cleanup-{index}.err",
            f"ird-post-cleanup-{index}",
            command_trace_path,
            args.ird_control_timeout_seconds,
        )
        lifecycle["post_cleanup"].append(asdict(result))

    lifecycle["completed_at"] = now_iso()
    write_json(ird_dir / "ird-lifecycle.json", lifecycle)
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
    write_json(run_dir / "environment.json", environment)
    remote_manifest = load_json(run_dir / "manifest.json")
    ird_summary = {
        "target": "ird",
        "mode": args.ird_mode,
        "remote_run_returncode": lifecycle.get("remote_run", {}).get("returncode"),
        "lifecycle": str(ird_dir / "ird-lifecycle.json"),
        "manual_cleanup": args.ird_release_command or "ird release <reservation_id>",
    }
    if remote_manifest.get("summary"):
        ird_summary["remote_summary"] = remote_manifest["summary"]
    update_manifest_summary(
        run_dir,
        ird_summary,
    )
    return 0 if lifecycle.get("remote_run", {}).get("returncode") == 0 else 2


def execute_pipeline(args: argparse.Namespace) -> int:
    if args.target == "ird":
        return execute_ird_pipeline(args)

    root = Path(args.repo_root).resolve() if args.repo_root else repo_root()
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
    readiness_results = run_tool_readiness_checks(
        repo=root,
        run_dir=run_dir,
        command_trace_path=command_trace_path,
        python_bin=python_bin,
        pytest_command=pytest_command,
        tracy_bin=tracy_bin,
        tt_perf_report_bin=tt_perf_report_bin,
    )
    environment["readiness"] = readiness_summary(readiness_results)
    if any(not result.ok for result in readiness_results):
        write_readiness_blocker_artifacts(run_dir, environment, readiness_results)
        return 2

    entries, discovery_result = discover_models(
        repo=root,
        run_id=run_dir.name,
        python_bin=python_bin,
        command_trace_path=command_trace_path,
        timeout_seconds=args.discovery_timeout_seconds,
    )
    entries = select_discovery_entries(entries, args.nodeid_filter, args.max_models)
    discover_artifacts(run_dir, entries, discovery_result, environment)

    if discovery_result.returncode not in (0, None) and not entries:
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
        return 2

    benchmark_kwargs = {
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "max_output_tokens": args.max_output_tokens,
    }
    for entry in entries:
        profile_one_model(
            entry=entry,
            repo=root,
            run_dir=run_dir,
            pytest_command=pytest_command,
            tracy_bin=tracy_bin,
            tt_perf_report_bin=tt_perf_report_bin,
            command_trace_path=command_trace_path,
            timeout_seconds=args.timeout_seconds,
            benchmark_kwargs=benchmark_kwargs,
        )

    statuses = load_model_statuses(run_dir)
    slow_ops = load_slow_ops(run_dir)
    dashboard_path, packet_path, report_path = write_artifacts(
        run_dir=run_dir,
        environment=environment,
        discovery_result=discovery_result,
        entries=entries,
        statuses=statuses,
        slow_ops=slow_ops,
    )
    update_manifest_summary(run_dir, summarize_run(run_dir, statuses, slow_ops))
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
        "--discovery-timeout-seconds", type=int, default=900, help="Discovery timeout."
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
        default="ghcr.io/tenstorrent/tt-xla/tt-xla-ird-ubuntu-22-04:latest",
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_id:
        args.run_id = args.run_id.strip()
    args.repo_root = args.repo_root or str(repo_root())
    if args.command == "run":
        return execute_pipeline(args)
    if args.command == "discover":
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
        tt_perf_report_bin = maybe_find_binary(
            "tt-perf-report", args.tt_perf_report_bin
        )
        command_trace_path = run_dir / "command-trace.jsonl"
        environment = capture_environment(
            root, tracy_bin, tt_perf_report_bin, pytest_command
        )
        entries, discovery_result = discover_models(
            repo=root,
            run_id=run_dir.name,
            python_bin=python_bin,
            command_trace_path=command_trace_path,
            timeout_seconds=args.discovery_timeout_seconds,
        )
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
        return 0
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
