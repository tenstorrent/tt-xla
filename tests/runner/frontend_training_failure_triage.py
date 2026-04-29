# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate bounded frontend training-failure triage bundles.

This tool turns a bounded set of training failures from the model test YAML into
review-ready frontend triage artifacts. It is intentionally planning-driven and
bounded: it focuses on rows whose current reason indicates that
``tt-forge-models`` does not implement ``unpack_forward_output`` for the model.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
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
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "artifacts" / "frontend_training_triage"
_TRAINING_UTILS_PATH = (
    _PROJECT_ROOT / "third_party" / "tt_forge_models" / "training_utils.py"
)
_UNPACK_REASON = (
    "tt-forge-models doesn't implement unpack_forward_output for this model."
)
_KNOWN_FRONTEND_SIGNATURES = (
    "unpack_forward_output",
    "deformable_im2col",
    "decoder_input_ids",
    "decoder_inputs_embeds",
    "Expected more than 1 value per channel",
    "targets required",
    "targets should not be none",
    "'NoneType' object has no attribute 'max'",
)
_SAFE_PATH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass
class TriageResult:
    test_id: str
    classification: str
    reason: str | None
    stale_reason_suspected: bool
    resolution: str
    next_manual_step: str
    loader_path: str
    loader_exists: bool
    has_custom_unpack_forward_output: bool
    training_utils_path: str
    output_dir: str
    draft_issue_path: str | None
    attempt_log_path: str | None


@dataclass
class TriageSummary:
    config_path: str
    output_root: str
    selected_test_count: int
    frontend_count: int
    draft_issue_count: int
    attempt_log_count: int
    stale_reason_suspected_count: int
    results: list[dict[str, Any]]


@dataclass
class ConfigRefreshRecommendation:
    test_id: str
    loader_path: str
    current_reason: str | None
    recommended_action: str
    rationale: str
    next_manual_step: str


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


def build_loader_path(project_root: Path, test_id: str) -> Path:
    parts = test_id.split("/")
    if len(parts) < 2:
        raise ValueError(f"Unsupported test_id format: {test_id}")

    model_path_parts = [
        validate_safe_path_component(part, "test_id model path") for part in parts[:-1]
    ]
    framework_and_variant = parts[-1]
    framework = validate_safe_path_component(
        framework_and_variant.split("-", 1)[0],
        "test_id framework",
    )
    models_root = (
        project_root.expanduser().resolve() / "third_party" / "tt_forge_models"
    ).resolve()
    loader_path = (
        models_root.joinpath(*model_path_parts) / framework / "loader.py"
    ).resolve()
    return require_path_within(loader_path, models_root, "loader path")


def loader_has_custom_unpack_forward_output(loader_path: Path) -> bool:
    if not loader_path.is_file():
        return False

    tree = ast.parse(loader_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelLoader":
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.FunctionDef)
                    and stmt.name == "unpack_forward_output"
                ):
                    return True
            return False
    return False


def classify_frontend_reason(reason: str | None) -> str:
    if not reason:
        return "no_draft_attempt_log"

    if reason == _UNPACK_REASON:
        return "frontend"

    if any(token in reason for token in _KNOWN_FRONTEND_SIGNATURES):
        return "frontend"

    return "no_draft_attempt_log"


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


def write_manifest(path: Path, result: TriageResult) -> None:
    path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_summary(path: Path, summary: TriageSummary) -> None:
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_review_report(path: Path, summary: TriageSummary) -> None:
    draft_rows = [result for result in summary.results if result["draft_issue_path"]]
    stale_rows = [
        result for result in summary.results if result["stale_reason_suspected"]
    ]
    attempt_rows = [result for result in summary.results if result["attempt_log_path"]]

    lines = [
        "# Frontend Training Failure Triage Review Report",
        "",
        "## Summary",
        f"- selected tests: `{summary.selected_test_count}`",
        f"- frontend-classified rows: `{summary.frontend_count}`",
        f"- draft issue packets: `{summary.draft_issue_count}`",
        f"- attempt logs: `{summary.attempt_log_count}`",
        f"- stale reason suspected: `{summary.stale_reason_suspected_count}`",
        "",
        "## Draft Candidates",
    ]

    if draft_rows:
        for row in draft_rows:
            lines.extend(
                [
                    f"- `{row['test_id']}`",
                    f"  - draft: `{row['draft_issue_path']}`",
                    f"  - loader: `{row['loader_path']}`",
                ]
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Stale Reason Suspected"])
    if stale_rows:
        for row in stale_rows:
            lines.extend(
                [
                    f"- `{row['test_id']}`",
                    f"  - resolution: `{row['resolution']}`",
                    f"  - next step: `{row['next_manual_step']}`",
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
                ]
            )
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_config_refresh_recommendations(
    path: Path, recommendations: list[ConfigRefreshRecommendation]
) -> None:
    payload = [asdict(recommendation) for recommendation in recommendations]
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_config_refresh_patch(
    path: Path, recommendations: list[ConfigRefreshRecommendation]
) -> None:
    test_config: dict[str, Any] = {}
    for recommendation in recommendations:
        test_config[recommendation.test_id] = {
            "reason": "REVIEW_NEEDED: stale frontend failure reason suspected",
            "notes": {
                "previous_reason": recommendation.current_reason,
                "recommended_action": recommendation.recommended_action,
                "rationale": recommendation.rationale,
                "next_manual_step": recommendation.next_manual_step,
            },
        }

    payload = {"test_config": test_config}
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def write_draft_issue(path: Path, result: TriageResult) -> None:
    body = f"""# Frontend Training Failure Triage Draft

## Summary
`{result.test_id}` currently fails in the bounded training cohort with a frontend-style signal:
`{result.reason}`

## Training-Failure Classification
- classification: `{result.classification}`
- likely owner: `tt-forge-models` model loader / training-helper review
- loader path: `{result.loader_path}`
- custom `unpack_forward_output` present: `{str(result.has_custom_unpack_forward_output).lower()}`

## Failure Signature And Repro Context
- bounded cohort reason: `{result.reason}`
- config source: `tests/runner/test_config/torch/test_config_training_single_device.yaml`
- workflow target: frontend issue analysis

## Evidence Links
- loader path: `{result.loader_path}`
- shared helper path: `{result.training_utils_path}`

## What Was Tried
- matched the training-failure row against the bounded frontend cohort rule
- resolved the corresponding `tt_forge_models` loader path from the test identifier
- inspected whether the loader implements `unpack_forward_output`

## Owner / Next Action Suggestion
- inspect whether the model should override `unpack_forward_output` in the loader
- if the output type is unambiguous and shared across models, consider extending `training_utils.py`
- keep filing draft-only until a human confirms the owner and proposed fix shape

## Open Questions / Handoff Notes
- does the model output need a model-specific unpacking rule or a shared handler registration?
- is CPU-only inspection required to confirm the expected training output tensor?
"""
    path.write_text(body, encoding="utf-8")


def write_attempt_log(path: Path, result: TriageResult) -> None:
    body = f"""workflow_path=frontend
test_id={result.test_id}
reason={result.reason}
loader_path={result.loader_path}
loader_exists={str(result.loader_exists).lower()}
custom_unpack_forward_output={str(result.has_custom_unpack_forward_output).lower()}
resolution={result.resolution}

No draft issue packet was generated.

Attempted steps:
1. Read bounded training config row from the provided YAML.
2. Classified the failure using the frontend signature rules.
3. Resolved the expected loader path from the test identifier.
4. Inspected the loader AST for a custom `unpack_forward_output` implementation.

Blocked decision:
- {result.resolution}

Next manual step:
- {result.next_manual_step}
"""
    path.write_text(body, encoding="utf-8")


def decide_resolution(
    *,
    classification: str,
    loader_exists: bool,
    has_custom_unpack_forward_output: bool,
) -> tuple[str, str]:
    if classification != "frontend":
        return (
            "frontend signature rule did not match strongly enough for a draft packet",
            "inspect the model loader and relevant runtime evidence before promoting this row to a review-ready draft.",
        )
    if not loader_exists:
        return (
            "expected tt_forge_models loader does not exist at the derived path",
            "confirm the loader location or reconcile the test identifier with the current tt_forge_models layout.",
        )
    if has_custom_unpack_forward_output:
        return (
            "loader already implements custom unpack_forward_output, so the YAML failure reason may be stale or the remaining failure is subtler than the bounded draft rule",
            "inspect the custom loader implementation and refresh the failing reason with current runtime evidence before drafting an issue.",
        )
    return (
        "bounded frontend draft rule matched: loader lacks custom unpack_forward_output and is a candidate for model/helper review",
        "review the generated draft issue and confirm whether the fix belongs in the model loader or shared training_utils handler registry.",
    )


def build_config_refresh_recommendation(
    result: TriageResult,
) -> ConfigRefreshRecommendation | None:
    if not result.stale_reason_suspected:
        return None

    return ConfigRefreshRecommendation(
        test_id=result.test_id,
        loader_path=result.loader_path,
        current_reason=result.reason,
        recommended_action="refresh_yaml_failure_reason",
        rationale=result.resolution,
        next_manual_step=result.next_manual_step,
    )


def triage_test_entry(
    *,
    project_root: Path,
    config_path: Path,
    test_id: str,
    entry: dict[str, Any],
    output_root: Path,
) -> TriageResult:
    reason = entry.get("reason")
    classification = classify_frontend_reason(reason)
    loader_path = build_loader_path(project_root, test_id)
    loader_exists = loader_path.is_file()
    has_custom_unpack = loader_has_custom_unpack_forward_output(loader_path)
    resolution, next_manual_step = decide_resolution(
        classification=classification,
        loader_exists=loader_exists,
        has_custom_unpack_forward_output=has_custom_unpack,
    )
    stale_reason_suspected = (
        classification == "frontend" and loader_exists and has_custom_unpack
    )

    output_dir = build_output_dir(output_root, test_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    draft_issue_path: Path | None = None
    attempt_log_path: Path | None = None
    should_write_draft = (
        classification == "frontend" and loader_exists and not has_custom_unpack
    )

    result = TriageResult(
        test_id=test_id,
        classification=classification,
        reason=reason,
        stale_reason_suspected=stale_reason_suspected,
        resolution=resolution,
        next_manual_step=next_manual_step,
        loader_path=str(loader_path),
        loader_exists=loader_exists,
        has_custom_unpack_forward_output=has_custom_unpack,
        training_utils_path=str(_TRAINING_UTILS_PATH),
        output_dir=str(output_dir),
        draft_issue_path=None,
        attempt_log_path=None,
    )

    write_manifest(output_dir / "manifest.json", result)

    if should_write_draft:
        draft_issue_path = output_dir / "draft_issue.md"
        write_draft_issue(draft_issue_path, result)
    else:
        attempt_log_path = output_dir / "attempt.log"
        write_attempt_log(attempt_log_path, result)

    result.draft_issue_path = str(draft_issue_path) if draft_issue_path else None
    result.attempt_log_path = str(attempt_log_path) if attempt_log_path else None
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
        if entry.get("reason") == _UNPACK_REASON:
            selected.append(test_id)
    return selected


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--test-id",
        action="append",
        default=[],
        help="Specific test id to triage. Repeat for multiple tests. Defaults to all bounded unpack_forward_output rows.",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    config = load_training_config(args.config)
    selected_tests = collect_selected_tests(config, args.test_id)
    results: list[TriageResult] = []

    for test_id in selected_tests:
        results.append(
            triage_test_entry(
                project_root=args.project_root,
                config_path=args.config,
                test_id=test_id,
                entry=config[test_id],
                output_root=output_root,
            )
        )

    summary = TriageSummary(
        config_path=str(args.config),
        output_root=str(output_root),
        selected_test_count=len(results),
        frontend_count=sum(
            1 for result in results if result.classification == "frontend"
        ),
        draft_issue_count=sum(
            1 for result in results if result.draft_issue_path is not None
        ),
        attempt_log_count=sum(
            1 for result in results if result.attempt_log_path is not None
        ),
        stale_reason_suspected_count=sum(
            1 for result in results if result.stale_reason_suspected
        ),
        results=[asdict(result) for result in results],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_summary(build_output_file(output_root, "summary.json"), summary)
    write_review_report(build_output_file(output_root, "review_report.md"), summary)
    recommendations = [
        recommendation
        for recommendation in (
            build_config_refresh_recommendation(result) for result in results
        )
        if recommendation is not None
    ]
    write_config_refresh_recommendations(
        build_output_file(output_root, "config_refresh_recommendations.json"),
        recommendations,
    )
    write_config_refresh_patch(
        build_output_file(output_root, "config_refresh_patch.yaml"),
        recommendations,
    )

    print(
        "Generated "
        f"{len(selected_tests)} frontend triage bundle(s) in {args.output_root} "
        f"(drafts={summary.draft_issue_count}, attempt_logs={summary.attempt_log_count}, "
        f"stale_reason_suspected={summary.stale_reason_suspected_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
