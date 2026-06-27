# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.bounded_training_workflow_validation import validate_record


def test_validate_record_passes_for_frontend_manifest():
    manifest = {
        "test_id": "yolov6/pytorch-N-single_device-training",
        "classification": "frontend",
        "reason": "frontend reason",
        "draft_issue_path": "/tmp/draft.md",
        "attempt_log_path": None,
        "next_manual_step": "review draft",
    }
    record = validate_record(manifest, "frontend")
    assert record.binary_result == "pass"
    assert record.failed_conditions == []


def test_validate_record_passes_for_runtime_manifest():
    manifest = {
        "test_id": "pointpillars/pytorch-pointpillars-single_device-training",
        "bringup_status": "FAILED_RUNTIME",
        "classification": "draft_issue",
        "owner_hint": "tt-metal",
        "reason": "runtime reason",
        "draft_issue_path": "/tmp/draft.md",
        "attempt_log_path": None,
        "next_manual_step": "capture debug logs",
    }
    record = validate_record(manifest, "runtime")
    assert record.binary_result == "pass"
    assert record.failed_conditions == []


def test_validate_record_fails_when_evidence_missing():
    manifest = {
        "test_id": "stable_diffusion_unet/pytorch-Base-single_device-training",
        "classification": "frontend",
        "reason": "frontend reason",
        "draft_issue_path": None,
        "attempt_log_path": None,
        "next_manual_step": "inspect logs",
    }
    record = validate_record(manifest, "frontend")
    assert record.binary_result == "error"
    assert "evidence bundle link is missing" in record.failed_conditions
