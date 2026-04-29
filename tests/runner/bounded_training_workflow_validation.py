# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Validate bounded frontend/runtime workflow outputs against the PRD-006 rubric."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationRecord:
    test_id: str
    workflow_path: str
    binary_result: str
    evidence_path: str
    failed_conditions: list[str]
    corrective_steps: list[str]
    failure_taxonomy: str | None


@dataclass
class ValidationSummary:
    frontend_output_root: str
    runtime_output_root: str
    total_items: int
    pass_count: int
    fail_count: int
    error_count: int
    records: list[dict[str, Any]]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(manifest: dict[str, Any], workflow_path: str) -> ValidationRecord:
    failed_conditions: list[str] = []
    corrective_steps: list[str] = []

    if not manifest.get("test_id"):
        failed_conditions.append("missing cohort identity")
        corrective_steps.append("ensure manifest.json records the exact test_id for the cohort row")

    if workflow_path == "frontend":
        path_explicit = manifest.get("classification") == "frontend"
    else:
        path_explicit = bool(manifest.get("owner_hint"))
    if not path_explicit:
        failed_conditions.append("workflow path or owner classification is not explicit")
        corrective_steps.append("record explicit workflow-path classification in the manifest")

    evidence_present = bool(manifest.get("draft_issue_path") or manifest.get("attempt_log_path"))
    if not evidence_present:
        failed_conditions.append("evidence bundle link is missing")
        corrective_steps.append("emit either draft_issue.md or attempt.log and persist its path in the manifest")

    outcome_explicit = bool(manifest.get("draft_issue_path") or manifest.get("attempt_log_path"))
    if not outcome_explicit:
        failed_conditions.append("recommendation or reduction outcome is not explicit")
        corrective_steps.append("record whether the item produced a draft packet or an attempt log")

    blockers_present = False
    if workflow_path == "frontend":
        blockers_present = bool(manifest.get("next_manual_step"))
    else:
        blockers_present = bool(manifest.get("next_manual_step"))
    if not blockers_present:
        failed_conditions.append("remaining blocker or next manual step is missing")
        corrective_steps.append("record the next manual step in the manifest")

    if workflow_path == "frontend":
        contract_ok = all(
            key in manifest
            for key in (
                "test_id",
                "classification",
                "reason",
                "draft_issue_path",
                "attempt_log_path",
                "next_manual_step",
            )
        )
    else:
        contract_ok = all(
            key in manifest
            for key in (
                "test_id",
                "bringup_status",
                "classification",
                "owner_hint",
                "reason",
                "draft_issue_path",
                "attempt_log_path",
                "next_manual_step",
            )
        )
    if not contract_ok:
        failed_conditions.append("output does not match the shared packet contract")
        corrective_steps.append("populate the required manifest fields for the workflow path")

    if failed_conditions:
        if not evidence_present:
            binary_result = "error"
            failure_taxonomy = "precondition_violation"
        else:
            binary_result = "fail"
            failure_taxonomy = "goal_not_achieved"
    else:
        binary_result = "pass"
        failure_taxonomy = None

    return ValidationRecord(
        test_id=manifest.get("test_id", ""),
        workflow_path=workflow_path,
        binary_result=binary_result,
        evidence_path=manifest.get("draft_issue_path") or manifest.get("attempt_log_path") or "",
        failed_conditions=failed_conditions,
        corrective_steps=corrective_steps,
        failure_taxonomy=failure_taxonomy,
    )


def collect_manifests(output_root: Path) -> list[Path]:
    return sorted(output_root.glob("*/manifest.json"))


def write_summary(path: Path, summary: ValidationSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_review_packet(path: Path, summary: ValidationSummary) -> None:
    lines = [
        "# Bounded Training Workflow Validation Review Packet",
        "",
        "## Summary",
        f"- total items: `{summary.total_items}`",
        f"- pass: `{summary.pass_count}`",
        f"- fail: `{summary.fail_count}`",
        f"- error: `{summary.error_count}`",
        "",
        "## Per-Item Results",
    ]
    for record in summary.records:
        lines.extend(
            [
                f"- `{record['test_id']}`",
                f"  - workflow: `{record['workflow_path']}`",
                f"  - result: `{record['binary_result']}`",
                f"  - evidence: `{record['evidence_path']}`",
            ]
        )
        if record["failed_conditions"]:
            lines.append(f"  - failed conditions: `{'; '.join(record['failed_conditions'])}`")
            lines.append(f"  - corrective steps: `{'; '.join(record['corrective_steps'])}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-output-root", type=Path, required=True)
    parser.add_argument("--runtime-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    records: list[ValidationRecord] = []

    for manifest_path in collect_manifests(args.frontend_output_root):
        records.append(validate_record(load_json(manifest_path), "frontend"))
    for manifest_path in collect_manifests(args.runtime_output_root):
        records.append(validate_record(load_json(manifest_path), "runtime"))

    summary = ValidationSummary(
        frontend_output_root=str(args.frontend_output_root),
        runtime_output_root=str(args.runtime_output_root),
        total_items=len(records),
        pass_count=sum(1 for record in records if record.binary_result == "pass"),
        fail_count=sum(1 for record in records if record.binary_result == "fail"),
        error_count=sum(1 for record in records if record.binary_result == "error"),
        records=[asdict(record) for record in records],
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_root / "validation_summary.json", summary)
    write_review_packet(args.output_root / "validation_review_packet.md", summary)

    print(
        "Validated "
        f"{summary.total_items} bounded workflow item(s) "
        f"(pass={summary.pass_count}, fail={summary.fail_count}, error={summary.error_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
