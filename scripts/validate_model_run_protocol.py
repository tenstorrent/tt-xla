#!/usr/bin/env python3
"""Validate the repository-owned model-run job protocol package."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/protocols/model-run-job-protocol.md"
CLAUDE_SKILL = ROOT / ".claude/skills/ttxla-model-run-job/SKILL.md"
CODEX_SKILL = ROOT / ".codex/skills/ttxla-model-run-job/SKILL.md"
LAUNCH_TEMPLATE = ROOT / "docs/protocols/model-run-job-launch-record-template.md"
BLOCKER_TEMPLATE = ROOT / "docs/protocols/model-run-job-blocker-template.md"
NORMALIZED_TEMPLATE = ROOT / "docs/protocols/model-run-job-normalized-results-template.csv"

REQUIRED_FILES = (
    PROTOCOL,
    CLAUDE_SKILL,
    CODEX_SKILL,
    LAUNCH_TEMPLATE,
    BLOCKER_TEMPLATE,
    NORMALIZED_TEMPLATE,
)

PROTOCOL_REQUIRED_TEXT = (
    "Agent-specific skills must reference this protocol",
    "nvidia_validation",
    "tt_model_validation",
    "collectability_check",
    "--nvidia-cohort-json",
    "human_review_required_before_external_post",
    "model-run-job-launch-record-template.md",
    "model-run-job-blocker-template.md",
    "model-run-job-normalized-results-template.csv",
    "Do not post GitHub comments",
)

WRAPPER_REQUIRED_TEXT = (
    "docs/protocols/model-run-job-protocol.md",
    "Read it first",
    "Do not fork or restate launch logic",
)

WRAPPER_FORBIDDEN_TEXT = (
    "pytest",
    "nvidia-smi",
    "test_models",
    "--nvidia-cohort-json",
    "ssh ",
    "gh issue",
    "gh api",
)

NORMALIZED_HEADER = [
    "run_id",
    "job_type",
    "manifest_path",
    "manifest_sha256",
    "test_case_id",
    "model_id",
    "task",
    "framework",
    "param",
    "status_raw",
    "outcome_normalized",
    "outcome_class",
    "evidence_path",
    "pytest_log",
    "junit_xml",
    "started_at_utc",
    "ended_at_utc",
    "duration_seconds",
    "runner_host",
    "repo_ref",
    "submodule_refs",
    "notes",
]


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_text(path: Path) -> str:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def require_contains(name: str, text: str, required: tuple[str, ...]) -> None:
    missing = [needle for needle in required if needle not in text]
    if missing:
        fail(f"{name} missing required text: {', '.join(missing)}")


def require_not_contains(name: str, text: str, forbidden: tuple[str, ...]) -> None:
    present = [needle for needle in forbidden if needle in text]
    if present:
        fail(f"{name} contains wrapper-forbidden launch logic: {', '.join(present)}")


def validate_wrappers() -> None:
    for path in (CLAUDE_SKILL, CODEX_SKILL):
        text = read_text(path)
        rel = str(path.relative_to(ROOT))
        require_contains(rel, text, WRAPPER_REQUIRED_TEXT)
        require_not_contains(rel, text, WRAPPER_FORBIDDEN_TEXT)


def validate_protocol() -> None:
    text = read_text(PROTOCOL)
    require_contains(str(PROTOCOL.relative_to(ROOT)), text, PROTOCOL_REQUIRED_TEXT)


def validate_templates() -> None:
    launch = read_text(LAUNCH_TEMPLATE)
    blocker = read_text(BLOCKER_TEMPLATE)

    require_contains(
        str(LAUNCH_TEMPLATE.relative_to(ROOT)),
        launch,
        (
            "`run_id`",
            "`manifest_sha256`",
            "`repo_ref`",
            "`runner_host`",
            "`artifact_root`",
            "`approved_external_posts`",
        ),
    )
    require_contains(
        str(BLOCKER_TEMPLATE.relative_to(ROOT)),
        blocker,
        (
            "`failure_taxonomy`: `precondition_violation`",
            "`duplicate_dispatch_risk`: `true`",
            "`external_reporting_allowed`: `false`",
            "`resume_condition`",
        ),
    )

    if not NORMALIZED_TEMPLATE.exists():
        fail(f"missing required file: {NORMALIZED_TEMPLATE.relative_to(ROOT)}")
    with NORMALIZED_TEMPLATE.open(newline="", encoding="utf-8") as csv_file:
        header = next(csv.reader(csv_file), None)
    if header != NORMALIZED_HEADER:
        fail("normalized results template header does not match protocol contract")


def main() -> int:
    for path in REQUIRED_FILES:
        if not path.exists():
            fail(f"missing required file: {path.relative_to(ROOT)}")

    validate_protocol()
    validate_wrappers()
    validate_templates()
    print("model-run protocol validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
