# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from tests.runner.runtime_training_failure_reduction import (
    build_output_dir,
    build_pytest_node_id,
    build_rerun_command,
    classify_runtime_entry,
    collect_selected_tests,
    derive_python_bin,
    extract_debug_evidence,
    find_debug_log,
    probe_rerun_environment,
    reduce_test_entry,
)


def write_executable_script(path: Path, body: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(body)


def test_build_pytest_node_id_uses_training_test_id():
    assert (
        build_pytest_node_id("pointpillars/pytorch-pointpillars-single_device-training")
        == "tests/runner/test_models.py::test_all_models_torch[pointpillars/pytorch-pointpillars-single_device-training]"
    )


def test_build_rerun_command_uses_pytest_and_node_id():
    command = build_rerun_command(
        "/tmp/pytest-bin",
        "pointpillars/pytorch-pointpillars-single_device-training",
    )
    assert command == [
        "/tmp/pytest-bin",
        "-vv",
        "-s",
        "tests/runner/test_models.py::test_all_models_torch[pointpillars/pytorch-pointpillars-single_device-training]",
    ]


def test_derive_python_bin_from_pytest_path():
    tmp_bin = Path("/tmp/env/bin")
    # derive_python_bin only resolves when the sibling python path exists.
    # Use a temp layout in the next tests for behavior proof.
    assert derive_python_bin("/tmp/env/bin/pytest") is None


def test_probe_rerun_environment_reports_missing_modules(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    write_executable_script(
        fake_python,
        "\n".join(
            [
                "#!/bin/sh",
                "echo \"MISSING::psutil: No module named 'psutil'\"",
                "echo \"MISSING::torch: No module named 'torch'\"",
            ]
        )
        + "\n",
    )
    fake_pytest = bin_dir / "pytest"
    write_executable_script(fake_pytest, "#!/bin/sh\nexit 0\n")
    result = probe_rerun_environment(str(fake_pytest))
    assert result is not None
    assert "psutil" in result
    assert "torch" in result


def test_probe_rerun_environment_ignores_non_missing_warning_output(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    write_executable_script(
        fake_python,
        "\n".join(
            [
                "#!/bin/sh",
                'echo "WARNING:root:Defaulting to PJRT_DEVICE=CPU"',
            ]
        )
        + "\n",
    )
    fake_pytest = bin_dir / "pytest"
    write_executable_script(fake_pytest, "#!/bin/sh\nexit 0\n")
    assert probe_rerun_environment(str(fake_pytest)) is None


def test_classify_runtime_entry_tt_metal_for_memory_signal():
    classification, owner_hint, reduction_signal = classify_runtime_entry(
        "FAILED_RUNTIME",
        "Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 2003264 B which is beyond max L1 size of 1499136 B",
    )
    assert classification == "draft_issue"
    assert owner_hint == "tt-metal"
    assert "tt-metal ownership" in reduction_signal


def test_classify_runtime_entry_tt_alchemist_for_generic_runtime_signal():
    classification, owner_hint, reduction_signal = classify_runtime_entry(
        "FAILED_RUNTIME",
        "RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source.",
    )
    assert classification == "draft_issue"
    assert owner_hint == "tt-alchemist"
    assert "reduction-worthy" in reduction_signal


def test_classify_runtime_entry_attempt_log_for_hang():
    classification, owner_hint, reduction_signal = classify_runtime_entry(
        "FAILED_RUNTIME",
        "RuntimeError: Test Hangs",
    )
    assert classification == "attempt_log"
    assert owner_hint == "unknown"
    assert "hang" in reduction_signal


def test_classify_runtime_entry_uses_debug_evidence_for_hang(tmp_path: Path):
    log_path = tmp_path / "hang.log"
    log_path.write_text(
        "INFO Executing operation: ttnn.matmul\nRuntimeError: Test Hangs\n",
        encoding="utf-8",
    )
    debug_evidence = extract_debug_evidence(log_path)
    classification, owner_hint, reduction_signal = classify_runtime_entry(
        "FAILED_RUNTIME",
        "RuntimeError: Test Hangs",
        debug_evidence,
    )
    assert classification == "draft_issue"
    assert owner_hint == "tt-alchemist"
    assert "executing-operation evidence" in reduction_signal


def test_collect_selected_tests_defaults_to_runtime_rows():
    config = {
        "a": {"bringup_status": "FAILED_RUNTIME", "reason": "RuntimeError: Test Hangs"},
        "b": {"bringup_status": "FAILED_FE_COMPILATION", "reason": "frontend"},
        "c": {"reason": "RuntimeError: TT_FATAL conv2d"},
    }
    assert collect_selected_tests(config, []) == ["a", "c"]


def test_find_debug_log_matches_sanitized_test_id(tmp_path: Path):
    log_path = tmp_path / "pointpillars_pytorch-pointpillars-single_device-training.log"
    log_path.write_text("Executing operation: ttnn.matmul\n", encoding="utf-8")
    resolved = find_debug_log(
        tmp_path, "pointpillars/pytorch-pointpillars-single_device-training"
    )
    assert resolved == log_path


def test_runtime_build_output_dir_keeps_sanitized_id_under_output_root(tmp_path: Path):
    output_root = tmp_path / "artifacts"
    output_dir = build_output_dir(output_root, "../../escape")
    output_dir.relative_to(output_root.resolve())


def test_extract_debug_evidence_pulls_runtime_signals(tmp_path: Path):
    log_path = tmp_path / "runtime.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO Executing operation: ttnn.conv2d",
                "RuntimeError: TT_FATAL conv2d device failure",
                "%0 = ttnn.conv2d %arg0, %arg1 : tensor<...>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = extract_debug_evidence(log_path)
    assert evidence is not None
    assert evidence.executing_operation_lines == [
        "INFO Executing operation: ttnn.conv2d"
    ]
    assert "TT_FATAL" in evidence.runtime_signal_lines[0]
    assert "ttnn.conv2d" in evidence.ttnn_mlir_lines[0]


def test_reduce_test_entry_writes_tt_metal_draft(tmp_path: Path):
    result = reduce_test_entry(
        test_id="pointpillars/pytorch-pointpillars-single_device-training",
        entry={
            "bringup_status": "FAILED_RUNTIME",
            "reason": "Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 2003264 B which is beyond max L1 size of 1499136 B",
        },
        output_root=tmp_path / "artifacts",
    )
    assert result.owner_hint == "tt-metal"
    assert result.draft_issue_path is not None
    assert Path(result.draft_issue_path).is_file()


def test_reduce_test_entry_attaches_debug_evidence_when_present(tmp_path: Path):
    debug_root = tmp_path / "logs"
    debug_root.mkdir()
    log_path = debug_root / "densenet_pytorch-121_Xray-single_device-training.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO Executing operation: ttnn.add",
                "RuntimeError: Index put requires the source and destination dtypes match",
                "%0 = ttnn.add %arg0, %arg1 : tensor<...>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = reduce_test_entry(
        test_id="densenet/pytorch-121_Xray-single_device-training",
        entry={
            "bringup_status": "FAILED_RUNTIME",
            "reason": "RuntimeError: dtype mismatch",
        },
        output_root=tmp_path / "artifacts",
        debug_log_root=debug_root,
    )
    assert result.runtime_debug_captured is True
    assert result.debug_log_path == str(log_path)
    assert result.debug_evidence_path is not None
    assert Path(result.debug_evidence_path).is_file()
    assert "trim to the smallest repro-worthy snippet" in result.next_manual_step


def test_reduce_test_entry_can_attempt_bounded_rerun(tmp_path: Path):
    fake_pytest = tmp_path / "fake_pytest.sh"
    write_executable_script(
        fake_pytest,
        "\n".join(
            [
                "#!/bin/sh",
                'echo "INFO Executing operation: ttnn.add"',
                'echo "RuntimeError: Index put requires the source and destination dtypes match"',
                'echo "%0 = ttnn.add %arg0, %arg1 : tensor<...>"',
            ]
        )
        + "\n",
    )
    result = reduce_test_entry(
        test_id="densenet/pytorch-121_Xray-single_device-training",
        entry={
            "bringup_status": "FAILED_RUNTIME",
            "reason": "RuntimeError: dtype mismatch",
        },
        output_root=tmp_path / "artifacts",
        execute_rerun=True,
        pytest_bin=str(fake_pytest),
        rerun_timeout_sec=5,
    )
    assert result.rerun_attempted is True
    assert result.rerun_command is not None
    assert result.rerun_log_path is not None
    assert result.runtime_debug_captured is True
    assert result.debug_evidence_path is not None
    assert Path(result.rerun_log_path).is_file()
    assert result.owner_hint == "tt-alchemist"


def test_reduce_test_entry_can_force_run_skipped_rows_for_bounded_debug(tmp_path: Path):
    fake_pytest = tmp_path / "fake_pytest.sh"
    write_executable_script(
        fake_pytest,
        "\n".join(
            [
                "#!/bin/sh",
                'echo "force=$TT_XLA_FORCE_RUN_SKIPPED_TEST_IDS"',
                'echo "INFO Executing operation: ttnn.matmul"',
                'echo "RuntimeError: Test Hangs"',
                'echo "%0 = ttnn.matmul %arg0, %arg1 : tensor<...>"',
            ]
        )
        + "\n",
    )
    test_id = "stable_diffusion_unet/pytorch-Base-single_device-training"
    result = reduce_test_entry(
        test_id=test_id,
        entry={
            "status": "NOT_SUPPORTED_SKIP",
            "bringup_status": "FAILED_RUNTIME",
            "reason": "RuntimeError: Test Hangs",
        },
        output_root=tmp_path / "artifacts",
        execute_rerun=True,
        pytest_bin=str(fake_pytest),
        rerun_timeout_sec=5,
        force_run_skipped=True,
    )
    assert result.force_run_skipped is True
    assert result.rerun_attempted is True
    assert result.runtime_debug_captured is True
    assert result.debug_evidence_path is not None
    assert result.draft_issue_path is not None
    assert result.owner_hint == "tt-alchemist"
    assert result.rerun_log_path is not None
    assert f"force={test_id}" in Path(result.rerun_log_path).read_text(encoding="utf-8")


def test_reduce_test_entry_surfaces_rerun_precondition_failure(tmp_path: Path):
    fake_pytest = tmp_path / "fake_pytest_fail.sh"
    write_executable_script(
        fake_pytest,
        "\n".join(
            [
                "#!/bin/sh",
                "echo \"ImportError while loading conftest '/tmp/tests/conftest.py'.\"",
                "echo \"E   ModuleNotFoundError: No module named 'psutil'\"",
                "exit 4",
            ]
        )
        + "\n",
    )
    result = reduce_test_entry(
        test_id="densenet/pytorch-121_Xray-single_device-training",
        entry={
            "bringup_status": "FAILED_RUNTIME",
            "reason": "RuntimeError: dtype mismatch",
        },
        output_root=tmp_path / "artifacts",
        execute_rerun=True,
        pytest_bin=str(fake_pytest),
        rerun_timeout_sec=5,
    )
    assert result.rerun_attempted is True
    assert result.classification == "attempt_log"
    assert result.owner_hint == "unknown"
    assert result.attempt_log_path is not None
    assert "precondition violation" in result.reduction_signal
    assert "missing pytest/runtime dependencies" in result.next_manual_step


def test_reduce_test_entry_fast_fails_on_environment_preflight(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    write_executable_script(
        fake_python,
        "#!/bin/sh\necho \"MISSING::torch: No module named 'torch'\"\n",
    )
    fake_pytest = bin_dir / "pytest"
    write_executable_script(fake_pytest, "#!/bin/sh\necho should-not-run\nexit 99\n")
    result = reduce_test_entry(
        test_id="densenet/pytorch-121_Xray-single_device-training",
        entry={
            "bringup_status": "FAILED_RUNTIME",
            "reason": "RuntimeError: dtype mismatch",
        },
        output_root=tmp_path / "artifacts",
        execute_rerun=True,
        pytest_bin=str(fake_pytest),
        rerun_timeout_sec=5,
    )
    assert result.rerun_attempted is False
    assert result.rerun_returncode is None
    assert result.classification == "attempt_log"
    assert "torch" in result.reduction_signal


def test_reduce_test_entry_surfaces_cpu_fallback_or_skip_as_attempt_log(tmp_path: Path):
    fake_pytest = tmp_path / "fake_pytest_cpu_skip.sh"
    write_executable_script(
        fake_pytest,
        "\n".join(
            [
                "#!/bin/sh",
                'echo "WARNING:root:Defaulting to PJRT_DEVICE=CPU"',
                'echo "tests/runner/test_models.py::test_all_models_torch[densenet/pytorch-121_Xray-single_device-training] SKIPPED"',
                "exit 0",
            ]
        )
        + "\n",
    )
    result = reduce_test_entry(
        test_id="densenet/pytorch-121_Xray-single_device-training",
        entry={
            "bringup_status": "FAILED_RUNTIME",
            "reason": "RuntimeError: dtype mismatch",
        },
        output_root=tmp_path / "artifacts",
        execute_rerun=True,
        pytest_bin=str(fake_pytest),
        rerun_timeout_sec=5,
    )
    assert result.rerun_attempted is True
    assert result.classification == "attempt_log"
    assert result.owner_hint == "unknown"
    assert "PJRT_DEVICE=CPU" in result.reduction_signal
    assert "TT hardware" in result.next_manual_step


def test_reduce_test_entry_surfaces_timeout_as_attempt_log(tmp_path: Path):
    fake_pytest = tmp_path / "fake_pytest_timeout.sh"
    write_executable_script(
        fake_pytest,
        "\n".join(
            [
                "#!/bin/sh",
                'echo "Running tests/runner/test_models.py::test_all_models_torch[stable_diffusion_unet/pytorch-Base-single_device-training]"',
                'echo "TIMEOUT after 240s"',
                "exit 0",
            ]
        )
        + "\n",
    )
    result = reduce_test_entry(
        test_id="stable_diffusion_unet/pytorch-Base-single_device-training",
        entry={
            "bringup_status": "FAILED_RUNTIME",
            "reason": "RuntimeError: Test Hangs",
        },
        output_root=tmp_path / "artifacts",
        execute_rerun=True,
        pytest_bin=str(fake_pytest),
        rerun_timeout_sec=5,
        force_run_skipped=True,
    )
    assert result.rerun_attempted is True
    assert result.classification == "attempt_log"
    assert "timed out" in result.reduction_signal
    assert "narrower debug run" in result.next_manual_step


def test_reduce_test_entry_writes_attempt_log_for_hang(tmp_path: Path):
    result = reduce_test_entry(
        test_id="beit/image_classification/pytorch-Base_Patch16_224-single_device-training",
        entry={
            "bringup_status": "FAILED_RUNTIME",
            "reason": "RuntimeError: Test Hangs",
        },
        output_root=tmp_path / "artifacts",
    )
    assert result.owner_hint == "unknown"
    assert result.attempt_log_path is not None
    assert Path(result.attempt_log_path).is_file()
