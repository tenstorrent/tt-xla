# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_pipeline_module():
    module_path = (
        Path(__file__).resolve().parent / "scripts" / "ttxla_profile_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("ttxla_profile_pipeline", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_profile_status(run_dir):
    return {
        "model": {
            "model_identity": "model-a",
            "nodeid": "tests/benchmark/test_llms.py::test_llama_3_2_1b",
            "source_path": "tests/benchmark/test_llms.py",
        },
        "terminal_state": "passed",
        "profile_status": "passed",
        "model_status": "passed",
        "taxonomy": "validated_pass",
        "reason": "",
        "next_action": "Review dashboard rankings and choose the next optimization target.",
        "status_path": str(run_dir / "profiles" / "model-a" / "status.json"),
        "artifacts": {
            "ir_dir": str(run_dir / "profiles" / "model-a" / "ir"),
            "tt_perf_report": str(
                run_dir / "profiles" / "model-a" / "perf-report" / "tt-perf-report.txt"
            ),
            "copied_ir_count": 1,
        },
        "stages": {
            "tt_perf_report": {"state": "generated"},
            "ir": {"state": "collected"},
        },
        "slow_ops": str(run_dir / "profiles" / "model-a" / "slow-ops.json"),
    }


def sample_slow_ops(run_dir, status):
    return [
        {
            "global_rank": 1,
            "model_identity": "model-a",
            "op_name": "matmul",
            "op_type": "compute",
            "duration_us": 120.5,
            "profile_status": "passed",
            "model_status": "passed",
            "taxonomy": "validated_pass",
            "status_path": str(run_dir / "profiles" / "model-a" / "status.json"),
            "ir_dir": status["artifacts"]["ir_dir"],
            "perf_report": status["artifacts"]["tt_perf_report"],
        }
    ]


def sample_discovery_entry(pipeline):
    return pipeline.DiscoveryEntry(
        run_identity="run-5009-demo-0001",
        nodeid="tests/benchmark/test_llms.py::test_llama_3_2_1b",
        source_path="tests/benchmark/test_llms.py",
        test_name="test_llama_3_2_1b",
        benchmark_family="llm",
        model_identity="model-a",
        artifact_slug="tests_benchmark_test_llms_py_test_llama_3_2_1b",
    )


def sample_discovery_result(pipeline, tmp_path):
    return pipeline.CommandResult(
        stage="discover",
        command=["python", "-m", "pytest", "--collect-only"],
        cwd=str(tmp_path),
        returncode=0,
        timed_out=False,
        start_time="2026-06-01T00:00:00+00:00",
        end_time="2026-06-01T00:00:01+00:00",
        duration_seconds=1.0,
        stdout_path=str(tmp_path / "discover.out"),
        stderr_path=str(tmp_path / "discover.err"),
    )


def sample_environment(tmp_path):
    return {
        "repo_root": str(tmp_path),
        "hostname": "example-host",
        "python": "3.12",
        "git": {"sha": "deadbeef", "branch": "demo"},
    }


def write_report_fixture_files(pipeline, run_dir, status):
    (run_dir / "manifest.json").write_text(
        json.dumps({"run": {"run_id": "run-5009-demo"}, "summary": {}}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "model-manifest.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "run_identity": "run-5009-demo-0001",
                        "nodeid": "tests/benchmark/test_llms.py::test_llama_3_2_1b",
                        "source_path": "tests/benchmark/test_llms.py",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    status_path = Path(status["status_path"])
    pipeline.ensure_dir(status_path.parent)
    pipeline.ensure_dir(Path(status["artifacts"]["ir_dir"]))
    perf_report_path = Path(status["artifacts"]["tt_perf_report"])
    pipeline.ensure_dir(perf_report_path.parent)
    perf_report_path.write_text("tt-perf-report output\n", encoding="utf-8")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status_path


def write_sample_perf_csv(csv_path):
    csv_path.write_text(
        "\n".join(
            [
                "op_name,op_type,duration_us,model",
                "matmul,compute,120.5,model-a",
                "gelu,activation,15.0,model-a",
                "add,elementwise,8.5,model-b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def partial_finalize_entries():
    return [
        {
            "run_identity": "run-5009-partial-finalize-0001",
            "nodeid": "tests/benchmark/test_vision.py::test_mnist",
            "source_path": "tests/benchmark/test_vision.py",
            "test_name": "test_mnist",
            "benchmark_family": "vision",
            "model_identity": "test_mnist",
            "artifact_slug": "vision",
        },
        {
            "run_identity": "run-5009-partial-finalize-0002",
            "nodeid": "tests/benchmark/resnet_jax_benchmark.py::test_resnet_jax",
            "source_path": "tests/benchmark/resnet_jax_benchmark.py",
            "test_name": "test_resnet_jax",
            "benchmark_family": "jax",
            "model_identity": "test_resnet_jax",
            "artifact_slug": "jax",
        },
    ]


def write_partial_finalize_manifest(run_dir, repo, entries):
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run": {
                    "run_id": run_dir.name,
                    "created_at": "2026-06-02T22:00:00+00:00",
                    "repo_root": str(repo),
                    "run_dir": str(run_dir),
                    "status": "discovered",
                },
                "command": "python -m pytest --collect-only -q tests/benchmark/test_vision.py",
                "discovery": {"returncode": 0, "timed_out": False, "note": ""},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model-manifest.json").write_text(
        json.dumps({"models": entries}),
        encoding="utf-8",
    )


def write_partial_finalize_existing_status(pipeline, run_dir, entries):
    existing_dir = run_dir / "profiles" / "vision"
    pipeline.ensure_dir(existing_dir)
    (existing_dir / "slow-ops.json").write_text(
        json.dumps(
            {
                "model": "test_mnist",
                "nodeid": entries[0]["nodeid"],
                "rows": [],
                "summary": {"row_count": 0},
            }
        ),
        encoding="utf-8",
    )
    (existing_dir / "status.json").write_text(
        json.dumps(
            {
                "model": {
                    "model_identity": "test_mnist",
                    "nodeid": entries[0]["nodeid"],
                },
                "terminal_state": "blocked",
                "profile_status": "failed",
                "model_status": "failed",
                "taxonomy": "environment_failure",
                "status_path": str(existing_dir / "status.json"),
                "artifacts": {},
                "stages": {},
                "slow_ops": str(existing_dir / "slow-ops.json"),
            }
        ),
        encoding="utf-8",
    )


def assert_partial_finalize_outputs(pipeline, run_dir):
    statuses = pipeline.load_model_statuses(run_dir)
    assert len(statuses) == 2
    jax_status = json.loads(
        (run_dir / "profiles" / "jax" / "status.json").read_text(encoding="utf-8")
    )
    assert jax_status["taxonomy"] == "not_started"
    assert (run_dir / "dashboard.html").exists()
    assert (run_dir / "claude-report-packet.html").exists()
    assert (run_dir / "report.html").exists()


def assert_partial_finalize_requirements(run_dir):
    requirements = json.loads((run_dir / "requirements.json").read_text())
    reqs = {item["id"]: item["status"] for item in requirements["requirements"]}
    assert reqs["REQ-F-003"] == "passed"
    assert reqs["REQ-F-006"] == "passed"


def assert_sample_perf_csv(parsed):
    assert parsed["summary"]["row_count"] == 3
    assert parsed["rows"][0]["op_name"] == "matmul"
    assert parsed["summary"]["op_type_totals"]["compute"] == 120.5


def assert_dashboard_report_text(dashboard_path, packet_path, report_path):
    dashboard_text = dashboard_path.read_text(encoding="utf-8")
    packet_text = packet_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    assert "Search" in dashboard_text
    assert "Global slow operations" in dashboard_text
    assert "Model status" in dashboard_text
    assert "REQ-F-008" in packet_text
    assert "REQ-F-009" in report_text
    assert "dashboard.html" in report_text


def load_requirement_artifacts(run_dir):
    requirements = json.loads((run_dir / "requirements.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    requirements_by_id = {item["id"]: item for item in requirements["requirements"]}
    return requirements, manifest, requirements_by_id


def assert_requirement_summary(requirements, manifest):
    assert manifest["run"]["status"] == "completed"
    assert requirements["issue"]["number"] == 5009
    assert requirements["summary"]["total"] == 9


def assert_requirement_passed(requirements_by_id, requirement_id, expected_paths=None):
    requirement = requirements_by_id[requirement_id]
    assert requirement["status"] == "passed"
    for expected_path in expected_paths or []:
        assert str(expected_path) in requirement["evidence_paths"]


def assert_requirement_coverage(
    run_dir, status_path, dashboard_path, packet_path, report_path
):
    requirements, manifest, requirements_by_id = load_requirement_artifacts(run_dir)
    assert_requirement_summary(requirements, manifest)
    assert_requirement_passed(requirements_by_id, "REQ-F-006", [status_path])
    assert_requirement_passed(requirements_by_id, "REQ-F-007", [dashboard_path])
    assert_requirement_passed(
        requirements_by_id,
        "REQ-F-008",
        [packet_path, report_path],
    )
    assert_requirement_passed(requirements_by_id, "REQ-F-009")


def parse_ird_run_args(pipeline):
    return pipeline.build_parser().parse_args(
        [
            "--target",
            "ird",
            "--run-id",
            "run-5009-demo",
            "--ird-bin",
            "ird",
            "--ird-docker-image",
            "xla",
            "--ird-timeout",
            "0:10",
            "--readiness-timeout-seconds",
            "120",
            "--ird-cluster",
            "tt_aus",
            "--ird-team",
            "sw",
            "--ird-machine",
            "aus-wh-01",
            "--ird-num-pcie-chips",
            "1",
            "--ird-remote-repo-root",
            "/work/tt-xla",
            "--ird-remote-output-root",
            "/work/tt-xla/artifacts/prd-009/ttxla-profile",
            "--benchmark-file",
            "tests/benchmark/test_vision.py",
            "--nodeid-filter",
            "test_vision.py",
            "--max-models",
            "1",
            "run",
        ]
    )


def assert_ird_scheduler_command(command, remote_command):
    assert command[:4] == ["ird", "run", "wormhole_b0", "--docker-image"]
    assert command[4] == "xla"
    assert "--cluster" in command
    assert "--team" in command
    assert "--machine" in command
    assert remote_command == command[-1]


def assert_remote_pipeline_command(remote_command):
    expected_fragments = [
        "--target local",
        "--readiness-timeout-seconds 120",
        "--max-raw-artifact-bytes 100000000",
        "--run-budget-seconds 300",
        "--benchmark-file tests/benchmark/test_vision.py",
        "--nodeid-filter test_vision.py",
        "--max-models 1",
        "ttxla_profile_pipeline.py",
        "run",
    ]
    for fragment in expected_fragments:
        assert fragment in remote_command


def test_parse_collect_output_discovers_benchmark_nodes():
    pipeline = load_pipeline_module()
    output = "\n".join(
        [
            "tests/benchmark/test_llms.py::test_llama_3_2_1b",
            "tests/benchmark/test_encoders.py::test_bert",
            "tests/benchmark/test_vision.py::test_resnet50",
            "tests/benchmark/resnet_jax_benchmark.py::test_resnet_jax",
            "4 tests collected in 0.12s",
        ]
    )

    entries = pipeline.parse_collect_output(output, "run-5009-demo")

    assert [entry.nodeid for entry in entries] == [
        "tests/benchmark/test_llms.py::test_llama_3_2_1b",
        "tests/benchmark/test_encoders.py::test_bert",
        "tests/benchmark/test_vision.py::test_resnet50",
        "tests/benchmark/resnet_jax_benchmark.py::test_resnet_jax",
    ]
    assert entries[0].run_identity == "run-5009-demo-0001"
    assert entries[0].source_path == "tests/benchmark/test_llms.py"
    assert entries[0].benchmark_family == "llm"
    assert entries[-1].benchmark_family == "jax"


def test_select_discovery_entries_filters_and_limits_nodes():
    pipeline = load_pipeline_module()
    entries = pipeline.parse_collect_output(
        "\n".join(
            [
                "tests/benchmark/test_llms.py::test_llama_3_2_1b",
                "tests/benchmark/test_vision.py::test_resnet50",
                "tests/benchmark/test_vision.py::test_mobilenet",
            ]
        ),
        "run-5009-demo",
    )

    selected = pipeline.select_discovery_entries(entries, ["test_vision.py"], 1)

    assert [entry.nodeid for entry in selected] == [
        "tests/benchmark/test_vision.py::test_resnet50"
    ]


def test_selected_benchmark_files_defaults_or_resolves_requested_paths(tmp_path):
    pipeline = load_pipeline_module()

    assert pipeline.selected_benchmark_files(tmp_path, []) == pipeline.benchmark_files(
        tmp_path
    )
    assert pipeline.selected_benchmark_files(
        tmp_path, ["tests/benchmark/test_vision.py"]
    ) == [tmp_path / "tests/benchmark/test_vision.py"]


def test_infer_taxonomy_distinguishes_environment_model_and_terminal_states():
    pipeline = load_pipeline_module()

    taxonomy, reason = pipeline.infer_taxonomy(0, False, "", {"model": "demo"}, True)
    assert taxonomy == "validated_pass"
    assert "perf report" in reason

    taxonomy, reason = pipeline.infer_taxonomy(
        1, False, "ModuleNotFoundError: no module named tracy", {}, False
    )
    assert taxonomy == "environment_failure"
    assert "environment" in reason

    taxonomy, reason = pipeline.infer_taxonomy(
        1, False, "AssertionError: PCC comparison failed", {"measurements": [{}]}, False
    )
    assert taxonomy == "validated_fail"
    assert "validation failed" in reason

    taxonomy, reason = pipeline.infer_taxonomy(1, True, "", {}, False)
    assert taxonomy == "not_started"
    assert "timed out" in reason


def test_status_semantics_do_not_treat_profile_success_as_model_success():
    pipeline = load_pipeline_module()
    text = "\n".join(
        [
            "2026-06-02 | critical | Always | TT_FATAL: Inputs to matmul must be tilized",
            "RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13",
        ]
    )

    assert pipeline.infer_profile_status(0, False) == "passed"
    assert pipeline.infer_model_status(0, False, text, {}) == "failed"

    taxonomy, reason = pipeline.infer_taxonomy(
        returncode=0,
        timed_out=False,
        text=text,
        benchmark_json={},
        perf_report_ok=True,
    )

    assert taxonomy == "model_failure"
    assert "model or runtime behavior failed" in reason


def test_skip_detection_ignores_unrelated_log_words():
    pipeline = load_pipeline_module()
    text = "nanobind: leaked types!\\n - ... skipped remainder"

    assert pipeline.infer_model_status(0, False, text, {"model": "demo"}) == "passed"

    taxonomy, _ = pipeline.infer_taxonomy(
        returncode=0,
        timed_out=False,
        text=text,
        benchmark_json={"model": "demo"},
        perf_report_ok=False,
    )
    assert taxonomy == "pipeline_error"


def test_model_success_ignores_environment_warning_text():
    pipeline = load_pipeline_module()
    text = "Profiler API not found for PJRT plugin"

    assert pipeline.infer_model_status(0, False, text, {"model": "demo"}) == "passed"


def test_discovery_and_manifest_artifacts_are_written(tmp_path):
    pipeline = load_pipeline_module()
    run_dir = tmp_path / "artifacts" / "prd-009" / "ttxla-profile" / "run-5009-demo"
    pipeline.ensure_dir(run_dir)

    entry = pipeline.DiscoveryEntry(
        run_identity="run-5009-demo-0001",
        nodeid="tests/benchmark/test_llms.py::test_llama_3_2_1b",
        source_path="tests/benchmark/test_llms.py",
        test_name="test_llama_3_2_1b",
        benchmark_family="llm",
        model_identity="test_llama_3_2_1b",
        artifact_slug="tests_benchmark_test_llms_py_test_llama_3_2_1b",
    )
    discovery_result = pipeline.CommandResult(
        stage="discover",
        command=["python", "-m", "pytest", "--collect-only"],
        cwd=str(tmp_path),
        returncode=0,
        timed_out=False,
        start_time="2026-06-01T00:00:00+00:00",
        end_time="2026-06-01T00:00:01+00:00",
        duration_seconds=1.0,
        stdout_path=str(tmp_path / "discover.out"),
        stderr_path=str(tmp_path / "discover.err"),
    )
    environment = {
        "repo_root": str(tmp_path),
        "hostname": "example-host",
        "python": "3.12",
        "git": {"sha": "deadbeef", "branch": "demo"},
    }

    pipeline.discover_artifacts(run_dir, [entry], discovery_result, environment)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    model_manifest = json.loads((run_dir / "model-manifest.json").read_text())
    assert manifest["counts"]["discovered_models"] == 1
    assert model_manifest["models"][0]["nodeid"] == entry.nodeid
    assert (run_dir / "environment.json").exists()


def test_parse_perf_csv_and_render_dashboard(tmp_path):
    pipeline = load_pipeline_module()
    csv_path = tmp_path / "ops_perf_results.csv"
    write_sample_perf_csv(csv_path)

    parsed = pipeline.parse_perf_csv(csv_path, "model-a", "model-a")
    assert_sample_perf_csv(parsed)

    run_dir = tmp_path / "run"
    pipeline.ensure_dir(run_dir)
    status = sample_profile_status(run_dir)
    slow_ops = sample_slow_ops(run_dir, status)
    status_path = write_report_fixture_files(pipeline, run_dir, status)

    dashboard_path, packet_path, report_path = pipeline.write_artifacts(
        run_dir=run_dir,
        environment=sample_environment(tmp_path),
        discovery_result=sample_discovery_result(pipeline, tmp_path),
        entries=[sample_discovery_entry(pipeline)],
        statuses=[status],
        slow_ops=slow_ops,
    )

    assert_dashboard_report_text(dashboard_path, packet_path, report_path)
    assert_requirement_coverage(
        run_dir, status_path, dashboard_path, packet_path, report_path
    )


def test_run_subprocess_records_missing_command(tmp_path):
    pipeline = load_pipeline_module()
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    trace_path = tmp_path / "command-trace.jsonl"

    result = pipeline.run_subprocess(
        command=["definitely-not-a-command-xyz"],
        cwd=tmp_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stage="profile",
        command_trace_path=trace_path,
        timeout_seconds=5,
    )

    assert result.returncode == 127
    assert "failed to start command" in result.note
    assert stderr_path.read_text(encoding="utf-8").strip()
    assert trace_path.exists()


def test_profile_command_accepts_python_module_tracy_invocation(tmp_path):
    pipeline = load_pipeline_module()

    command = pipeline.profile_command(
        tracy_bin="python3 -m tracy",
        pytest_command="pytest",
        nodeid="tests/benchmark/test_llms.py::test_llama_3_2_1b",
        profile_dir=tmp_path / "profile",
        benchmark_output=tmp_path / "benchmark.json",
        benchmark_args=[
            "--batch-size",
            "1",
            "--num-layers",
            "1",
            "--max-output-tokens",
            "3",
        ],
    )

    assert command[:3] == ["python3", "-m", "tracy"]
    assert "-m" in command
    assert "pytest" in command
    assert "--batch-size" in command
    assert "--num-layers" in command
    assert "--max-output-tokens" in command


def test_benchmark_args_are_routed_by_family():
    pipeline = load_pipeline_module()
    kwargs = {"batch_size": 2, "num_layers": 3, "max_output_tokens": 4}

    llm = pipeline.DiscoveryEntry(
        run_identity="run-5009-demo-0001",
        nodeid="tests/benchmark/test_llms.py::test_llama_3_2_1b",
        source_path="tests/benchmark/test_llms.py",
        test_name="test_llama_3_2_1b",
        benchmark_family="llm",
        model_identity="test_llama_3_2_1b",
        artifact_slug="llm",
    )
    encoder = pipeline.DiscoveryEntry(
        run_identity="run-5009-demo-0002",
        nodeid="tests/benchmark/test_encoders.py::test_bert",
        source_path="tests/benchmark/test_encoders.py",
        test_name="test_bert",
        benchmark_family="encoder",
        model_identity="test_bert",
        artifact_slug="encoder",
    )
    vision = pipeline.DiscoveryEntry(
        run_identity="run-5009-demo-0003",
        nodeid="tests/benchmark/test_vision.py::test_mnist",
        source_path="tests/benchmark/test_vision.py",
        test_name="test_mnist",
        benchmark_family="vision",
        model_identity="test_mnist",
        artifact_slug="vision",
    )
    jax = pipeline.DiscoveryEntry(
        run_identity="run-5009-demo-0004",
        nodeid="tests/benchmark/resnet_jax_benchmark.py::test_resnet_jax",
        source_path="tests/benchmark/resnet_jax_benchmark.py",
        test_name="test_resnet_jax",
        benchmark_family="jax",
        model_identity="test_resnet_jax",
        artifact_slug="jax",
    )

    assert pipeline.benchmark_args_for_entry(llm, kwargs) == [
        "--batch-size",
        "2",
        "--num-layers",
        "3",
        "--max-output-tokens",
        "4",
    ]
    assert pipeline.benchmark_args_for_entry(encoder, kwargs) == [
        "--num-layers",
        "3",
    ]
    assert pipeline.benchmark_args_for_entry(vision, kwargs) == []
    assert pipeline.benchmark_args_for_entry(jax, kwargs) == []


def test_prune_large_raw_artifacts_removes_only_oversized_tracy_files(tmp_path):
    pipeline = load_pipeline_module()
    profile_dir = tmp_path / "profile"
    logs_dir = profile_dir / "tracy" / ".logs"
    report_dir = profile_dir / "tracy" / "reports" / "2026_06_02_21_25_05"
    pipeline.ensure_dir(logs_dir)
    pipeline.ensure_dir(report_dir)
    large_raw = logs_dir / "tracy_ops_times.csv"
    small_raw = logs_dir / "tracy_ops_data.csv"
    large_report = report_dir / "profile_log_device.csv"
    large_raw.write_bytes(b"x" * 12)
    small_raw.write_bytes(b"x" * 4)
    large_report.write_bytes(b"x" * 11)

    pruned = pipeline.prune_large_raw_artifacts(profile_dir, max_bytes=10)

    assert {Path(item["path"]).name for item in pruned} == {
        "tracy_ops_times.csv",
        "profile_log_device.csv",
    }
    assert not large_raw.exists()
    assert small_raw.exists()
    assert not large_report.exists()


def test_terminalize_missing_model_statuses_writes_blocker_artifacts(tmp_path):
    pipeline = load_pipeline_module()
    run_dir = tmp_path / "run-5009-terminalize"
    pipeline.ensure_dir(run_dir)
    entries = [
        pipeline.DiscoveryEntry(
            run_identity="run-5009-terminalize-0001",
            nodeid="tests/benchmark/test_vision.py::test_mnist",
            source_path="tests/benchmark/test_vision.py",
            test_name="test_mnist",
            benchmark_family="vision",
            model_identity="test_mnist",
            artifact_slug="vision",
        ),
        pipeline.DiscoveryEntry(
            run_identity="run-5009-terminalize-0002",
            nodeid="tests/benchmark/resnet_jax_benchmark.py::test_resnet_jax",
            source_path="tests/benchmark/resnet_jax_benchmark.py",
            test_name="test_resnet_jax",
            benchmark_family="jax",
            model_identity="test_resnet_jax",
            artifact_slug="jax",
        ),
    ]
    existing_dir = run_dir / "profiles" / "vision"
    pipeline.ensure_dir(existing_dir)
    (existing_dir / "status.json").write_text(
        json.dumps(
            {
                "model": {"model_identity": "test_mnist"},
                "terminal_state": "blocked",
                "profile_status": "failed",
                "model_status": "failed",
                "taxonomy": "environment_failure",
                "status_path": str(existing_dir / "status.json"),
                "artifacts": {},
                "stages": {},
                "slow_ops": str(existing_dir / "slow-ops.json"),
            }
        ),
        encoding="utf-8",
    )

    created = pipeline.terminalize_missing_model_statuses(
        run_dir=run_dir,
        entries=entries,
        repo=tmp_path,
        reason="pipeline finalized before this discovered model emitted status.json",
    )

    assert len(created) == 1
    status = json.loads(
        (run_dir / "profiles" / "jax" / "status.json").read_text(encoding="utf-8")
    )
    assert status["profile_status"] == "not_run"
    assert status["model_status"] == "not_run"
    assert status["taxonomy"] == "not_started"
    assert status["terminal_state"] == "blocked"
    assert "before this discovered model emitted status.json" in status["reason"]
    assert (run_dir / "profiles" / "jax" / "slow-ops.json").exists()


def test_finalize_partial_run_from_manifest_renders_reports(tmp_path):
    pipeline = load_pipeline_module()
    run_dir = tmp_path / "run-5009-partial-finalize"
    pipeline.ensure_dir(run_dir)
    entries = partial_finalize_entries()
    write_partial_finalize_manifest(run_dir, tmp_path, entries)
    write_partial_finalize_existing_status(pipeline, run_dir, entries)

    finalized = pipeline.finalize_partial_run_from_manifest(
        run_dir=run_dir,
        repo=tmp_path,
        environment={
            "repo_root": str(tmp_path),
            "hostname": "example-host",
            "python": "3.12",
            "git": {"sha": "deadbeef", "branch": "demo"},
        },
        reason="outer IRD wrapper finalized the partial run",
    )

    assert finalized is True
    assert_partial_finalize_outputs(pipeline, run_dir)
    assert_partial_finalize_requirements(run_dir)


def test_run_records_readiness_blocker_when_tracy_is_missing(tmp_path):
    pipeline = load_pipeline_module()
    fake_perf_report = tmp_path / "tt-perf-report"
    fake_perf_report.write_text(
        "#!/bin/sh\necho tt-perf-report 1.2.4\n", encoding="utf-8"
    )
    fake_perf_report.chmod(0o755)
    run_dir = tmp_path / "run-5009-readiness-blocked"

    exit_code = pipeline.main(
        [
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            str(run_dir),
            "--tracy-bin",
            "definitely-not-tracy",
            "--tt-perf-report-bin",
            str(fake_perf_report),
            "run",
        ]
    )

    assert exit_code == 2
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    trace = (run_dir / "command-trace.jsonl").read_text(encoding="utf-8")

    assert manifest["run"]["status"] == "environment_blocked"
    assert manifest["summary"]["readiness"]["ok"] is False
    assert "readiness-tracy" in trace
    assert "definitely-not-tracy" in trace
    assert environment["readiness"]["failed"][0]["stage"] == "readiness-tracy"


def test_ird_run_command_wraps_remote_pipeline():
    pipeline = load_pipeline_module()
    args = parse_ird_run_args(pipeline)

    remote_command = pipeline.build_remote_pipeline_command(args, "run-5009-demo")
    command = pipeline.build_ird_run_command(args, remote_command)

    assert_ird_scheduler_command(command, remote_command)
    assert_remote_pipeline_command(remote_command)


def test_ird_remote_run_budget_uses_explicit_value():
    pipeline = load_pipeline_module()
    parser = pipeline.build_parser()
    args = parser.parse_args(
        [
            "--target",
            "ird",
            "--run-id",
            "run-5009-demo",
            "--ird-timeout",
            "1:00",
            "--run-budget-seconds",
            "42",
            "run",
        ]
    )

    remote_command = pipeline.build_remote_pipeline_command(args, "run-5009-demo")

    assert "--run-budget-seconds 42" in remote_command


def test_parse_ird_reservation_accepts_json_and_text():
    pipeline = load_pipeline_module()

    parsed_json = pipeline.parse_ird_reservation(
        '{"reservation_id": "58806", "target_host": "yyzc-wh-03:49668"}'
    )
    assert parsed_json.reservation_id == "58806"
    assert parsed_json.target_host == "yyzc-wh-03:49668"

    parsed_text = pipeline.parse_ird_reservation(
        "reservation id: 65671\nhost: aus-wh-01:12345\n"
    )
    assert parsed_text.reservation_id == "65671"
    assert parsed_text.target_host == "aus-wh-01:12345"


def test_ird_target_records_blocker_when_ird_binary_is_missing(tmp_path):
    pipeline = load_pipeline_module()
    run_dir = tmp_path / "run-5009-missing-ird"

    exit_code = pipeline.main(
        [
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            str(run_dir),
            "--target",
            "ird",
            "--ird-bin",
            "definitely-not-ird",
            "--ird-job-timeout-seconds",
            "5",
            "run",
        ]
    )

    assert exit_code == 2
    lifecycle_path = run_dir / "ird" / "ird-lifecycle.json"
    assert lifecycle_path.exists()
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    assert lifecycle["target"] == "ird"
    assert lifecycle["mode"] == "run"
    assert lifecycle["remote_run"]["returncode"] == 127
    assert "failed to start command" in lifecycle["remote_run"]["note"]
    trace = (run_dir / "command-trace.jsonl").read_text(encoding="utf-8")
    assert "definitely-not-ird" in trace


def test_ird_run_timeout_uses_nested_manifest_when_remote_pipeline_terminalized(
    tmp_path,
):
    pipeline = load_pipeline_module()
    root = tmp_path / "repo"
    root.mkdir()
    run_dir = tmp_path / "run-5009-ird-tail-timeout"
    parser = pipeline.build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(root),
            "--run-dir",
            str(run_dir),
            "--target",
            "ird",
            "--ird-bin",
            "ird",
            "--ird-job-timeout-seconds",
            "1",
            "run",
        ]
    )
    original_run_subprocess = pipeline.run_subprocess

    def fake_run_subprocess(**kwargs):
        assert kwargs["stage"] == "ird-run"
        pipeline.ensure_dir(run_dir)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run": {"completed_at": "2026-06-02T20:29:35+00:00"},
                    "summary": {"models": 1, "taxonomy": {"environment_failure": 1}},
                }
            ),
            encoding="utf-8",
        )
        return pipeline.CommandResult(
            stage="ird-run",
            command=kwargs["command"],
            cwd=str(root),
            returncode=-9,
            timed_out=True,
            start_time="2026-06-02T20:24:00+00:00",
            end_time="2026-06-02T20:34:00+00:00",
            duration_seconds=600,
            stdout_path=str(kwargs["stdout_path"]),
            stderr_path=str(kwargs["stderr_path"]),
            note="timed out after 1 seconds",
        )

    try:
        pipeline.run_subprocess = fake_run_subprocess
        exit_code = pipeline.execute_ird_pipeline(args)
    finally:
        pipeline.run_subprocess = original_run_subprocess

    assert exit_code == 0
    lifecycle = json.loads(
        (run_dir / "ird" / "ird-lifecycle.json").read_text(encoding="utf-8")
    )
    assert lifecycle["remote_run"]["timed_out"] is True
    assert lifecycle["remote_run"]["returncode"] == 0
    assert "nested manifest indicates" in lifecycle["remote_run"]["note"]


def test_ird_run_timeout_keeps_blocker_when_nested_manifest_failed_discovery(
    tmp_path,
):
    pipeline = load_pipeline_module()
    root = tmp_path / "repo"
    root.mkdir()
    run_dir = tmp_path / "run-5009-ird-discovery-failed"
    (run_dir / "manifest.json").parent.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run": {"completed_at": "2026-06-02T20:22:40+00:00"},
                "summary": {"discovery_failed": True, "returncode": 4},
            }
        ),
        encoding="utf-8",
    )

    returncode, note = pipeline.infer_nested_ird_returncode(run_dir)

    assert returncode == 2
    assert "discovery failure" in note


def test_ird_run_timeout_cancels_scheduler_job_when_nested_manifest_not_terminalized(
    tmp_path,
):
    pipeline = load_pipeline_module()
    root = tmp_path / "repo"
    root.mkdir()
    run_dir = tmp_path / "run-5009-ird-nonterminal-timeout"
    parser = pipeline.build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(root),
            "--run-dir",
            str(run_dir),
            "--target",
            "ird",
            "--ird-bin",
            "ird",
            "--ird-job-timeout-seconds",
            "1",
            "run",
        ]
    )
    original_run_subprocess = pipeline.run_subprocess

    def fake_run_subprocess(**kwargs):
        if kwargs["stage"] == "ird-run":
            pipeline.ensure_dir(kwargs["stderr_path"].parent)
            kwargs["stderr_path"].write_text(
                "Job submitted successfully with ID: 78328\n",
                encoding="utf-8",
            )
            return pipeline.CommandResult(
                stage="ird-run",
                command=kwargs["command"],
                cwd=str(root),
                returncode=-9,
                timed_out=True,
                start_time="2026-06-02T20:40:00+00:00",
                end_time="2026-06-02T20:49:00+00:00",
                duration_seconds=540,
                stdout_path=str(kwargs["stdout_path"]),
                stderr_path=str(kwargs["stderr_path"]),
                note="timed out after 1 seconds",
            )
        assert kwargs["stage"] == "ird-timeout-cleanup"
        assert kwargs["command"] == ["scancel", "78328"]
        return pipeline.CommandResult(
            stage="ird-timeout-cleanup",
            command=kwargs["command"],
            cwd=str(root),
            returncode=0,
            timed_out=False,
            start_time="2026-06-02T20:49:00+00:00",
            end_time="2026-06-02T20:49:01+00:00",
            duration_seconds=1,
            stdout_path=str(kwargs["stdout_path"]),
            stderr_path=str(kwargs["stderr_path"]),
        )

    try:
        pipeline.run_subprocess = fake_run_subprocess
        exit_code = pipeline.execute_ird_pipeline(args)
    finally:
        pipeline.run_subprocess = original_run_subprocess

    assert exit_code == 2
    lifecycle = json.loads(
        (run_dir / "ird" / "ird-lifecycle.json").read_text(encoding="utf-8")
    )
    assert lifecycle["remote_run"]["returncode"] == -9
    assert "requested scancel" in lifecycle["remote_run"]["note"]
    assert lifecycle["timeout_cleanup"]["job_id"] == "78328"
    assert lifecycle["timeout_cleanup"]["returncode"] == 0
