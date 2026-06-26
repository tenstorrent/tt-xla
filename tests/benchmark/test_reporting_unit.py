# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for the reporting helpers shared across benchmarks.

Pins the measurement-dict and output-JSON shapes that the llm and vllm drivers
share, so the extraction stays byte-identical to the inline code it replaced
(the vLLM path isn't runnable on CPU here).
"""

import json

from reporting import (
    aggregate_llm_decode_perf,
    throughput_measurement,
    ttft_measurement,
    write_benchmark_json,
)


def test_ttft_measurement_shape():
    assert ttft_measurement(12.5) == {
        "measurement_name": "ttft",
        "value": 12.5,
        "target": -1,
    }


def test_throughput_measurement_shape():
    assert throughput_measurement(3.0) == {
        "measurement_name": "samples_per_sec",
        "value": 3.0,
        "target": -1,
    }


def test_write_benchmark_json_stamps_and_dumps(tmp_path):
    results = {"model": "m", "config": {"existing": 1}}
    out = tmp_path / "out.json"

    write_benchmark_json(results, str(out), model_rawname="raw/name")

    # Stamping mutates the dict in place, then dumps it verbatim.
    assert results["project"] == "tt-forge/tt-xla"
    assert results["model_rawname"] == "raw/name"
    on_disk = json.loads(out.read_text())
    assert on_disk == results
    assert on_disk["config"]["existing"] == 1


def test_write_benchmark_json_custom_project(tmp_path):
    results = {}
    out = tmp_path / "out.json"
    write_benchmark_json(results, str(out), model_rawname="m", project="other/proj")
    assert json.loads(out.read_text())["project"] == "other/proj"


def test_write_benchmark_json_aggregates_perf_metrics(tmp_path, monkeypatch):
    # aggregate_ttnn_perf_metrics scans the cwd for "<base>*.json" files.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tt_xla_m_perf_metrics_0.json").write_text(
        json.dumps(
            {
                "summary": {
                    "total_ops": 10,
                    "total_shardable_ops": 4,
                    "effectively_sharded_ops": 2,
                    "system_memory_ops": 1,
                }
            }
        )
    )

    results = {"config": {}}
    out = tmp_path / "out.json"
    write_benchmark_json(
        results,
        str(out),
        model_rawname="m",
        ttnn_perf_metrics_file="tt_xla_m_perf_metrics",
    )

    cfg = json.loads(out.read_text())["config"]
    assert cfg["ttnn_total_ops"] == 10
    assert cfg["ttnn_effectively_sharded_ops"] == 2
    assert cfg["ttnn_effectively_sharded_percentage"] == 50.0
    assert cfg["ttnn_num_graphs"] == 1


def _write_perf_file(path, summary):
    path.write_text(json.dumps({"summary": summary}))


def test_aggregate_llm_decode_perf_uses_decode_graph(tmp_path, monkeypatch):
    # LLMs emit prefill (index 0) + decode (index 1); only decode should be used.
    monkeypatch.chdir(tmp_path)
    _write_perf_file(
        tmp_path / "tt_xla_m_perf_metrics_0.json",
        {"total_ops": 100, "total_shardable_ops": 50, "effectively_sharded_ops": 10},
    )
    _write_perf_file(
        tmp_path / "tt_xla_m_perf_metrics_1.json",
        {
            "total_ops": 7,
            "total_shardable_ops": 4,
            "effectively_sharded_ops": 3,
            "system_memory_ops": 1,
            "effectively_sharded_percentage": 75.0,
        },
    )

    results = {"config": {}}
    aggregate_llm_decode_perf("tt_xla_m_perf_metrics", results)

    cfg = results["config"]
    assert cfg["ttnn_total_ops"] == 7  # decode graph, not prefill's 100
    assert cfg["ttnn_total_shardable_ops"] == 4
    assert cfg["ttnn_effectively_sharded_ops"] == 3
    assert cfg["ttnn_system_memory_ops"] == 1
    assert cfg["ttnn_effectively_sharded_percentage"] == 75.0
    assert cfg["ttnn_num_graphs"] == 2


def test_aggregate_llm_decode_perf_unexpected_file_count(tmp_path, monkeypatch):
    # Anything other than exactly two files records the count and skips metrics.
    monkeypatch.chdir(tmp_path)
    _write_perf_file(tmp_path / "tt_xla_m_perf_metrics_0.json", {"total_ops": 5})

    results = {"config": {}}
    aggregate_llm_decode_perf("tt_xla_m_perf_metrics", results)

    assert results["config"] == {"ttnn_num_graphs": 1}
