# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

from tests.runner.prd003_doc_validator import CONTRACT_VERSION, run_validation


REQUIRED_RECORD_FIELDS = {
    "contract_version",
    "run_id",
    "record_id",
    "timestamp_utc",
    "repo",
    "ref",
    "surface_type",
    "surface_id",
    "flow_id",
    "environment",
    "status",
    "reason_code",
    "severity",
    "summary",
    "evidence",
    "doc_clarity",
    "actionability",
}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def test_prd003_doc_validator_emits_contract_records(tmp_path: Path):
    output_root = Path(os.getenv("TT_XLA_PRD003_OUTPUT_ROOT", tmp_path / "prd003_doc_validator"))
    run_id = os.getenv("TT_XLA_PRD003_RUN_ID", "prd003-doc-validator-pytest")
    target_id = os.getenv("TT_XLA_PRD003_TARGET_ID", "local")
    execute = _env_flag("TT_XLA_PRD003_EXECUTE")
    require_pass = _env_flag("TT_XLA_PRD003_REQUIRE_PASS")

    result = run_validation(
        output_root=output_root,
        run_id=run_id,
        target_id=target_id,
        execute=execute,
    )

    results_path = result.output_root / "records" / "jsonl" / "results.jsonl"
    summary_path = result.output_root / "summaries" / "coverage-summary.json"

    assert results_path.is_file()
    assert summary_path.is_file()
    assert result.summary["contract_version"] == CONTRACT_VERSION
    assert result.summary["record_count"] == 6

    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 6
    assert all(REQUIRED_RECORD_FIELDS <= record.keys() for record in records)
    assert all(record["contract_version"] == CONTRACT_VERSION for record in records)
    assert all(record["repo"] == "tt-xla" for record in records)
    assert all(record["evidence"]["command"] for record in records)

    if require_pass:
        assert result.summary["counts_by_status"] == {"pass": 6}
    else:
        assert result.summary["counts_by_status"] == {"fail": 6}
        assert all(record["reason_code"] == "doc-ambiguous-step" for record in records)
        assert all(record["doc_clarity"]["is_ambiguous"] for record in records)
