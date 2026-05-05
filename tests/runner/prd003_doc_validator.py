# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PRD-003 docs/install validation harness.

This runner emits PRD-003 contract records for the phase-1 docs/install
surfaces. It is intentionally fail-closed: manifest entries with unresolved
doc-clarity blockers are recorded as documentation failures instead of running
ambiguous package install, Docker, or source-build commands.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "prd003_doc_validator"


PHASE1_DOC_FLOWS: list[dict[str, Any]] = [
    {
        "flow_id": "CORPUS-DOCKER-001",
        "surface_id": "docker-container-start",
        "surface_type": "doc",
        "priority": "P0",
        "source_mode": "documented",
        "source_path": "docs/src/getting_started.md",
        "source_section": "Using a Docker Container to Run an Example",
        "command": (
            "docker run -it --rm --device /dev/tenstorrent "
            "-v /dev/hugepages-1G:/dev/hugepages-1G "
            "ghcr.io/tenstorrent/tt-xla-slim:latest"
        ),
        "expected_target": {
            "execution_target": "vm",
            "target_id": "local",
            "os_image": "ubuntu-22.04",
            "runtime": "docker",
        },
        "success_signal": {"type": "exit_zero", "value": ""},
        "doc_clarity": {
            "fail_closed": True,
            "blocker_ids": ["DOC-GAP-006"],
            "notes": (
                "Docker launch lacks a non-interactive readiness check and assumes "
                "TT device passthrough."
            ),
        },
    },
    {
        "flow_id": "CORPUS-WHEEL-001",
        "surface_id": "wheel-install-latest",
        "surface_type": "doc",
        "priority": "P0",
        "source_mode": "documented",
        "source_path": "docs/src/getting_started.md",
        "source_section": "Installing a Wheel and Running an Example",
        "command": (
            "pip install pjrt-plugin-tt "
            "--extra-index-url https://pypi.eng.aws.tenstorrent.com/"
        ),
        "expected_target": {
            "execution_target": "vm",
            "target_id": "local",
            "os_image": "ubuntu-22.04",
            "runtime": "python",
        },
        "success_signal": {"type": "exit_zero", "value": ""},
        "doc_clarity": {
            "fail_closed": True,
            "blocker_ids": ["DOC-GAP-003"],
            "notes": "Private index access and expected wheel version are not specified.",
        },
    },
    {
        "flow_id": "CORPUS-SOURCE-001",
        "surface_id": "source-system-deps",
        "surface_type": "doc",
        "priority": "P0",
        "source_mode": "documented",
        "source_path": "docs/src/getting_started.md",
        "source_section": "Building from Source",
        "command": (
            "sudo apt install protobuf-compiler libprotobuf-dev && "
            "sudo apt install ccache && "
            "sudo apt install libnuma-dev && "
            "sudo apt install libhwloc-dev && "
            "sudo apt install libboost-all-dev"
        ),
        "expected_target": {
            "execution_target": "vm",
            "target_id": "local",
            "os_image": "ubuntu-22.04",
            "runtime": "bash",
        },
        "success_signal": {"type": "exit_zero", "value": ""},
        "doc_clarity": {
            "fail_closed": True,
            "blocker_ids": ["DOC-GAP-003"],
            "notes": (
                "Required Python, Clang, GCC, Ninja, and CMake versions lack "
                "install commands."
            ),
        },
    },
    {
        "flow_id": "CORPUS-SOURCE-002",
        "surface_id": "source-clone",
        "surface_type": "doc",
        "priority": "P0",
        "source_mode": "documented",
        "source_path": "docs/src/getting_started.md",
        "source_section": "Building from Source",
        "command": "git clone https://github.com/tenstorrent/tt-xla.git",
        "expected_target": {
            "execution_target": "vm",
            "target_id": "local",
            "os_image": "ubuntu-22.04",
            "runtime": "git",
        },
        "success_signal": {"type": "artifact_exists", "value": "tt-xla"},
        "doc_clarity": {
            "fail_closed": True,
            "blocker_ids": ["DOC-GAP-003"],
            "notes": "Repo branch/ref is not pinned for repeatable validation.",
        },
    },
    {
        "flow_id": "CORPUS-SOURCE-004",
        "surface_id": "source-build",
        "surface_type": "doc",
        "priority": "P0",
        "source_mode": "documented",
        "source_path": "docs/src/getting_started.md",
        "source_section": "Building from Source",
        "command": "source venv/activate && cmake -G Ninja -B build && cmake --build build",
        "expected_target": {
            "execution_target": "vm",
            "target_id": "local",
            "os_image": "ubuntu-22.04",
            "runtime": "cmake",
        },
        "success_signal": {"type": "exit_zero", "value": ""},
        "doc_clarity": {
            "fail_closed": True,
            "blocker_ids": ["DOC-GAP-004"],
            "notes": "The creation and ownership of venv/activate are not documented.",
        },
    },
    {
        "flow_id": "CORPUS-SOURCE-006",
        "surface_id": "source-wheel-build",
        "surface_type": "doc",
        "priority": "P1",
        "source_mode": "documented",
        "source_path": "docs/src/getting_started.md",
        "source_section": "Building from Source",
        "command": "python setup.py bdist_wheel",
        "expected_target": {
            "execution_target": "vm",
            "target_id": "local",
            "os_image": "ubuntu-22.04",
            "runtime": "python",
        },
        "success_signal": {"type": "artifact_exists", "value": "dist/pjrt_plugin_tt*.whl"},
        "doc_clarity": {
            "fail_closed": True,
            "blocker_ids": ["DOC-GAP-004"],
            "notes": "Source-build environment setup is not fully specified before wheel build.",
        },
    },
]


@dataclass
class ValidationRun:
    run_id: str
    output_root: Path
    records: list[dict[str, Any]]
    summary: dict[str, Any]


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def redact(text: str) -> str:
    redacted = text
    for pattern, replacement in (
        (r"(?i)(token|password|secret|api[_-]?key)=\S+", r"\1=<redacted>"),
        (r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]+", r"\1<redacted>"),
    ):
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


def base_record(
    flow: dict[str, Any],
    run_id: str,
    output_root: Path,
    target_id: str,
) -> dict[str, Any]:
    flow_id = str(flow["flow_id"])
    environment = dict(flow["expected_target"])
    environment["target_id"] = target_id
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "record_id": f"{run_id}-{flow_id}",
        "timestamp_utc": utc_now(),
        "repo": "tt-xla",
        "ref": "local",
        "surface_type": flow["surface_type"],
        "surface_id": flow["surface_id"],
        "flow_id": flow_id,
        "environment": environment,
        "status": "error",
        "reason_code": "unknown-error",
        "severity": "medium",
        "summary": "",
        "evidence": {
            "doc_path": flow["source_path"],
            "doc_section": flow["source_section"],
            "command": flow["command"],
            "exit_code": None,
            "stdout_ref": str(output_root / "logs" / f"{flow_id}.stdout.log"),
            "stderr_ref": str(output_root / "logs" / f"{flow_id}.stderr.log"),
            "artifact_refs": [
                str(output_root / "snapshots" / "commands" / f"{flow_id}.command.txt")
            ],
        },
        "doc_clarity": {
            "is_ambiguous": False,
            "flags": [],
            "notes": "",
        },
        "actionability": {
            "owner_team": "tt-xla",
            "recommended_fix": "none",
            "next_step": "keep-monitoring",
        },
    }


def fail_closed_record(
    flow: dict[str, Any],
    run_id: str,
    output_root: Path,
    target_id: str,
) -> dict[str, Any]:
    record = base_record(flow, run_id, output_root, target_id)
    clarity = flow["doc_clarity"]
    record["status"] = "fail"
    record["reason_code"] = "doc-ambiguous-step"
    record["severity"] = "high"
    record["summary"] = (
        f"{flow['flow_id']} was not executed because the documented flow is ambiguous."
    )
    record["evidence"]["stdout_ref"] = None
    record["evidence"]["stderr_ref"] = None
    record["doc_clarity"] = {
        "is_ambiguous": True,
        "flags": list(clarity["blocker_ids"]),
        "notes": clarity["notes"],
    }
    record["actionability"] = {
        "owner_team": "tt-xla",
        "recommended_fix": (
            "Document prerequisites, pinned refs, success criteria, or accepted defaults "
            "before execution."
        ),
        "next_step": "resolve-doc-clarity-blocker",
    }
    return record


def classify_execution(
    record: dict[str, Any],
    flow: dict[str, Any],
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    record["evidence"]["exit_code"] = completed.returncode
    if completed.returncode != 0:
        record["status"] = "fail"
        record["reason_code"] = "broken-command"
        record["severity"] = "high"
        record["summary"] = f"{flow['flow_id']} exited non-zero."
        record["actionability"]["recommended_fix"] = "Fix the documented command or prerequisites."
        record["actionability"]["next_step"] = "open-docs-or-install-defect"
        return record

    signal = flow["success_signal"]
    signal_type = signal["type"]
    signal_value = signal["value"]
    matched = False
    if signal_type == "exit_zero":
        matched = True
    elif signal_type == "stdout_contains":
        matched = signal_value in completed.stdout
    elif signal_type == "stdout_regex":
        matched = re.search(signal_value, completed.stdout) is not None
    elif signal_type == "artifact_exists":
        matched = bool(signal_value)

    if matched:
        record["status"] = "pass"
        record["reason_code"] = "ok"
        record["severity"] = "low"
        record["summary"] = f"{flow['flow_id']} completed with expected success signal."
    else:
        record["status"] = "fail"
        record["reason_code"] = "assertion-failed"
        record["severity"] = "high"
        record["summary"] = (
            f"{flow['flow_id']} exited zero but did not match the expected success signal."
        )
        record["actionability"]["recommended_fix"] = "Align the command or expected success signal."
        record["actionability"]["next_step"] = "open-validation-defect"
    return record


def execute_flow(
    flow: dict[str, Any],
    run_id: str,
    output_root: Path,
    target_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    record = base_record(flow, run_id, output_root, target_id)
    flow_id = str(flow["flow_id"])
    stdout_path = output_root / "logs" / f"{flow_id}.stdout.log"
    stderr_path = output_root / "logs" / f"{flow_id}.stderr.log"
    try:
        completed = subprocess.run(
            flow["command"],
            shell=True,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        write_text(stdout_path, redact(exc.stdout or ""))
        write_text(stderr_path, redact(exc.stderr or ""))
        record["status"] = "error"
        record["reason_code"] = "timeout"
        record["summary"] = f"{flow_id} timed out after {timeout_seconds} seconds."
        record["evidence"]["exit_code"] = 124
        record["actionability"]["recommended_fix"] = (
            "Increase timeout or split the validation step."
        )
        record["actionability"]["next_step"] = "rerun-with-adjusted-timeout"
        return record

    write_text(stdout_path, redact(completed.stdout))
    write_text(stderr_path, redact(completed.stderr))
    return classify_execution(record, flow, completed)


def build_summary(run_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    counts_by_status: dict[str, int] = {}
    blockers: dict[str, int] = {}
    for record in records:
        counts_by_status[record["status"]] = counts_by_status.get(record["status"], 0) + 1
        for flag in record["doc_clarity"]["flags"]:
            blockers[flag] = blockers.get(flag, 0) + 1
    return {
        "run_id": run_id,
        "contract_version": CONTRACT_VERSION,
        "repo": "tt-xla",
        "ref": "local",
        "platform": platform.platform(),
        "record_count": len(records),
        "counts_by_status": counts_by_status,
        "doc_clarity_blockers": blockers,
        "top_recommended_next_steps": sorted(
            {record["actionability"]["next_step"] for record in records}
        ),
        "ended_at_utc": utc_now(),
    }


def run_validation(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = "prd003-doc-validator-pytest",
    target_id: str = "local",
    execute: bool = False,
    timeout_seconds: int = 1800,
) -> ValidationRun:
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    write_json(
        output_root / "manifest" / "tt-xla-prd003-manifest.lock.json",
        {
            "contract_version": CONTRACT_VERSION,
            "execute": execute,
            "platform": platform.platform(),
            "run_id": run_id,
            "target_id": target_id,
            "timestamp_utc": utc_now(),
        },
    )

    for flow in PHASE1_DOC_FLOWS:
        flow_id = str(flow["flow_id"])
        write_text(
            output_root / "snapshots" / "commands" / f"{flow_id}.command.txt",
            f"{flow['command']}\n",
        )
        if flow["doc_clarity"].get("fail_closed"):
            record = fail_closed_record(flow, run_id, output_root, target_id)
        elif execute:
            record = execute_flow(flow, run_id, output_root, target_id, timeout_seconds)
        else:
            record = base_record(flow, run_id, output_root, target_id)
            record["status"] = "error"
            record["reason_code"] = "infra-unavailable"
            record["summary"] = (
                f"{flow_id} was not executed because execute mode was not enabled."
            )
        records.append(record)
        write_json(output_root / "records" / "by-flow" / f"{flow_id}.result.json", record)

    jsonl_path = output_root / "records" / "jsonl" / "results.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )
    summary = build_summary(run_id, records)
    write_json(output_root / "summaries" / "coverage-summary.json", summary)
    return ValidationRun(run_id=run_id, output_root=output_root, records=records, summary=summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PRD-003 docs/install validation.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="prd003-doc-validator-cli")
    parser.add_argument("--target-id", default="local")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_validation(
        output_root=args.output_root,
        run_id=args.run_id,
        target_id=args.target_id,
        execute=args.execute,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
