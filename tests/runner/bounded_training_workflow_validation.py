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


_FRONTEND_REQUIRED_MANIFEST_KEYS = (
    "test_id",
    "classification",
    "reason",
    "draft_issue_path",
    "attempt_log_path",
    "next_manual_step",
)
_RUNTIME_REQUIRED_MANIFEST_KEYS = (
    "test_id",
    "bringup_status",
    "classification",
    "owner_hint",
    "reason",
    "draft_issue_path",
    "attempt_log_path",
    "next_manual_step",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_path(manifest: dict[str, Any]) -> str:
    return manifest.get("draft_issue_path") or manifest.get("attempt_log_path") or ""


def path_is_explicit(manifest: dict[str, Any], workflow_path: str) -> bool:
    if workflow_path == "frontend":
        return manifest.get("classification") == "frontend"
    return bool(manifest.get("owner_hint"))


def required_manifest_keys(workflow_path: str) -> tuple[str, ...]:
    if workflow_path == "frontend":
        return _FRONTEND_REQUIRED_MANIFEST_KEYS
    return _RUNTIME_REQUIRED_MANIFEST_KEYS


def record_failure(
    failed_conditions: list[str],
    corrective_steps: list[str],
    condition: str,
    corrective_step: str,
) -> None:
    failed_conditions.append(condition)
    corrective_steps.append(corrective_step)


def validate_record(manifest: dict[str, Any], workflow_path: str) -> ValidationRecord:
    failed_conditions: list[str] = []
    corrective_steps: list[str] = []
    output_evidence_path = evidence_path(manifest)

    if not manifest.get("test_id"):
        record_failure(
            failed_conditions,
            corrective_steps,
            "missing cohort identity",
            "ensure manifest.json records the exact test_id for the cohort row",
        )

    if not path_is_explicit(manifest, workflow_path):
        record_failure(
            failed_conditions,
            corrective_steps,
            "workflow path or owner classification is not explicit",
            "record explicit workflow-path classification in the manifest",
        )

    if not output_evidence_path:
        record_failure(
            failed_conditions,
            corrective_steps,
            "evidence bundle link is missing",
            "emit either draft_issue.md or attempt.log and persist its path in the manifest",
        )

    if not manifest.get("next_manual_step"):
        record_failure(
            failed_conditions,
            corrective_steps,
            "remaining blocker or next manual step is missing",
            "record the next manual step in the manifest",
        )

    if not all(key in manifest for key in required_manifest_keys(workflow_path)):
        record_failure(
            failed_conditions,
            corrective_steps,
            "output does not match the shared packet contract",
            "populate the required manifest fields for the workflow path",
        )

    if failed_conditions:
        if not output_evidence_path:
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
        evidence_path=output_evidence_path,
        failed_conditions=failed_conditions,
        corrective_steps=corrective_steps,
        failure_taxonomy=failure_taxonomy,
    )


def collect_manifests(output_root: Path) -> list[Path]:
    return sorted(output_root.glob("*/manifest.json"))


def validate_manifests(output_root: Path, workflow_path: str) -> list[ValidationRecord]:
    return [
        validate_record(load_json(manifest_path), workflow_path)
        for manifest_path in collect_manifests(output_root)
    ]


def count_records(records: list[ValidationRecord], binary_result: str) -> int:
    return sum(1 for record in records if record.binary_result == binary_result)


def build_validation_summary(
    *,
    frontend_output_root: Path,
    runtime_output_root: Path,
    records: list[ValidationRecord],
) -> ValidationSummary:
    return ValidationSummary(
        frontend_output_root=str(frontend_output_root),
        runtime_output_root=str(runtime_output_root),
        total_items=len(records),
        pass_count=count_records(records, "pass"),
        fail_count=count_records(records, "fail"),
        error_count=count_records(records, "error"),
        records=[asdict(record) for record in records],
    )


def write_summary(path: Path, summary: ValidationSummary) -> None:
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
            lines.append(
                f"  - failed conditions: `{'; '.join(record['failed_conditions'])}`"
            )
            lines.append(
                f"  - corrective steps: `{'; '.join(record['corrective_steps'])}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-output-root", type=Path, required=True)
    parser.add_argument("--runtime-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    records = [
        *validate_manifests(args.frontend_output_root, "frontend"),
        *validate_manifests(args.runtime_output_root, "runtime"),
    ]

    summary = build_validation_summary(
        frontend_output_root=args.frontend_output_root,
        runtime_output_root=args.runtime_output_root,
        records=records,
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
