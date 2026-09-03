# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from tests.runner.frontend_training_failure_triage import (
    _UNPACK_REASON,
    build_loader_path,
    build_output_dir,
    classify_frontend_reason,
    collect_selected_tests,
    load_training_config,
    loader_has_custom_unpack_forward_output,
    triage_test_entry,
)


def test_build_loader_path_for_nested_model_family(tmp_path: Path):
    loader_path = build_loader_path(
        tmp_path,
        "distilbert/question_answering/pytorch-Base_Cased_Distilled_Squad-single_device-training",
    )
    assert loader_path == (
        tmp_path
        / "third_party"
        / "tt_forge_models"
        / "distilbert"
        / "question_answering"
        / "pytorch"
        / "loader.py"
    )


def test_build_loader_path_rejects_path_traversal_component(tmp_path: Path):
    try:
        build_loader_path(
            tmp_path,
            "../question_answering/pytorch-Base_Cased_Distilled_Squad-single_device-training",
        )
    except ValueError as exc:
        assert "Unsafe test_id model path component" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsafe test_id path component")


def test_loader_has_custom_unpack_forward_output_detects_method(tmp_path: Path):
    loader_path = tmp_path / "loader.py"
    loader_path.write_text(
        """
class ModelLoader:
    def unpack_forward_output(self, output):
        return output
""",
        encoding="utf-8",
    )
    assert loader_has_custom_unpack_forward_output(loader_path) is True


def test_loader_has_custom_unpack_forward_output_false_when_missing(tmp_path: Path):
    loader_path = tmp_path / "loader.py"
    loader_path.write_text(
        """
class ModelLoader:
    def load_model(self):
        return None
""",
        encoding="utf-8",
    )
    assert loader_has_custom_unpack_forward_output(loader_path) is False


def test_collect_selected_tests_defaults_to_unpack_reason_only():
    config = {
        "a": {"reason": _UNPACK_REASON},
        "b": {"reason": "runtime failure"},
    }
    assert collect_selected_tests(config, []) == ["a"]


def test_classify_frontend_reason_recognizes_known_signature():
    assert classify_frontend_reason(_UNPACK_REASON) == "frontend"
    assert (
        classify_frontend_reason("missing decoder_input_ids in training") == "frontend"
    )
    assert classify_frontend_reason("L1 allocation failure") == "no_draft_attempt_log"


def test_load_training_config_requires_mapping(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")
    try:
        load_training_config(config_path)
    except ValueError as exc:
        assert "Expected mapping" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-mapping YAML")


def test_triage_test_entry_writes_draft_issue_for_missing_custom_unpack(tmp_path: Path):
    project_root = tmp_path / "repo"
    loader_path = (
        project_root
        / "third_party"
        / "tt_forge_models"
        / "yolov6"
        / "pytorch"
        / "loader.py"
    )
    loader_path.parent.mkdir(parents=True)
    loader_path.write_text(
        """
class ModelLoader:
    def load_model(self):
        return None
""",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"

    result = triage_test_entry(
        project_root=project_root,
        config_path=project_root / "config.yaml",
        test_id="yolov6/pytorch-N-single_device-training",
        entry={"reason": _UNPACK_REASON},
        output_root=output_root,
    )

    assert result.draft_issue_path is not None
    assert result.attempt_log_path is None
    draft_issue = Path(result.draft_issue_path)
    manifest = Path(result.output_dir) / "manifest.json"
    assert draft_issue.is_file()
    assert manifest.is_file()
    manifest_data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert manifest_data["classification"] == "frontend"
    assert manifest_data["has_custom_unpack_forward_output"] is False


def test_triage_test_entry_writes_attempt_log_when_custom_unpack_exists(tmp_path: Path):
    project_root = tmp_path / "repo"
    loader_path = (
        project_root
        / "third_party"
        / "tt_forge_models"
        / "maptr"
        / "pytorch"
        / "loader.py"
    )
    loader_path.parent.mkdir(parents=True)
    loader_path.write_text(
        """
class ModelLoader:
    def unpack_forward_output(self, output):
        return output
""",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"

    result = triage_test_entry(
        project_root=project_root,
        config_path=project_root / "config.yaml",
        test_id="maptr/pytorch-Tiny_R50_24e_Av2-single_device-training",
        entry={"reason": _UNPACK_REASON},
        output_root=output_root,
    )

    assert result.draft_issue_path is None
    assert result.attempt_log_path is not None
    attempt_log = Path(result.attempt_log_path)
    assert attempt_log.is_file()
    assert "No draft issue packet was generated." in attempt_log.read_text(
        encoding="utf-8"
    )


def test_build_output_dir_keeps_sanitized_id_under_output_root(tmp_path: Path):
    output_root = tmp_path / "artifacts"
    output_dir = build_output_dir(output_root, "../../escape")
    output_dir.relative_to(output_root.resolve())
