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
    assert taxonomy == "pending_terminalization"
    assert "timed out" in reason


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

    parsed = pipeline.parse_perf_csv(csv_path, "model-a", "model-a")
    assert parsed["summary"]["row_count"] == 3
    assert parsed["rows"][0]["op_name"] == "matmul"
    assert parsed["summary"]["op_type_totals"]["compute"] == 120.5

    run_dir = tmp_path / "run"
    pipeline.ensure_dir(run_dir)
    status = {
        "model": {
            "model_identity": "model-a",
            "nodeid": "tests/benchmark/test_llms.py::test_llama_3_2_1b",
            "source_path": "tests/benchmark/test_llms.py",
        },
        "terminal_state": "passed",
        "taxonomy": "validated_pass",
        "reason": "",
        "next_action": "Review dashboard rankings and choose the next optimization target.",
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
    slow_ops = [
        {
            "global_rank": 1,
            "model_identity": "model-a",
            "op_name": "matmul",
            "op_type": "compute",
            "duration_us": 120.5,
            "profile_status": "passed",
            "taxonomy": "validated_pass",
            "status_path": str(run_dir / "profiles" / "model-a" / "status.json"),
            "ir_dir": status["artifacts"]["ir_dir"],
            "perf_report": status["artifacts"]["tt_perf_report"],
        }
    ]

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

    dashboard_path, packet_path, report_path = pipeline.write_artifacts(
        run_dir=run_dir,
        environment={
            "repo_root": str(tmp_path),
            "hostname": "example-host",
            "python": "3.12",
            "git": {"sha": "deadbeef", "branch": "demo"},
        },
        discovery_result=pipeline.CommandResult(
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
        ),
        entries=[
            pipeline.DiscoveryEntry(
                run_identity="run-5009-demo-0001",
                nodeid="tests/benchmark/test_llms.py::test_llama_3_2_1b",
                source_path="tests/benchmark/test_llms.py",
                test_name="test_llama_3_2_1b",
                benchmark_family="llm",
                model_identity="model-a",
                artifact_slug="tests_benchmark_test_llms_py_test_llama_3_2_1b",
            )
        ],
        statuses=[status],
        slow_ops=slow_ops,
    )

    dashboard_text = dashboard_path.read_text(encoding="utf-8")
    packet_text = packet_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    assert "Search" in dashboard_text
    assert "Global slow operations" in dashboard_text
    assert "REQ-F-008" in packet_text
    assert "REQ-F-009" in report_text
    assert "dashboard.html" in report_text


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
        benchmark_kwargs={"batch_size": 1, "num_layers": 1, "max_output_tokens": 3},
    )

    assert command[:3] == ["python3", "-m", "tracy"]
    assert "-m" in command
    assert "pytest" in command


def test_run_records_readiness_blocker_when_tracy_is_missing(tmp_path):
    pipeline = load_pipeline_module()
    fake_perf_report = tmp_path / "tt-perf-report"
    fake_perf_report.write_text("#!/bin/sh\necho tt-perf-report 1.2.4\n", encoding="utf-8")
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
    parser = pipeline.build_parser()
    args = parser.parse_args(
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
            "--nodeid-filter",
            "test_vision.py",
            "--max-models",
            "1",
            "run",
        ]
    )

    remote_command = pipeline.build_remote_pipeline_command(args, "run-5009-demo")
    command = pipeline.build_ird_run_command(args, remote_command)

    assert command[:5] == ["ird", "run", "--docker-image", "xla", "--timeout"]
    assert "wormhole_b0" in command
    assert "--cluster" in command
    assert "--team" in command
    assert "--machine" in command
    assert remote_command == command[-1]
    assert "--target local" in remote_command
    assert "--nodeid-filter test_vision.py" in remote_command
    assert "--max-models 1" in remote_command
    assert "ttxla_profile_pipeline.py" in remote_command
    assert "run" in remote_command


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
