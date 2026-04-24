# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import torch

from tt_torch.backend import quetzal_analysis


class _FakeNode:
    def __init__(self, op_type: str):
        self.op_type = op_type


class _FakeGraph:
    def __init__(self, op_types: list[str]):
        self.nodes = {idx: _FakeNode(op_type) for idx, op_type in enumerate(op_types)}
        self.edges = [(idx, idx + 1, f"t{idx}") for idx in range(len(op_types) - 1)]


def test_get_quetzal_analysis_config_reads_env(monkeypatch):
    monkeypatch.setenv(quetzal_analysis.QUETZAL_ANALYSIS_PASSES_ENV, "all")
    monkeypatch.setenv(
        quetzal_analysis.QUETZAL_ANALYSIS_REPORT_PATH_ENV, "/tmp/quetzal-report"
    )

    config = quetzal_analysis.get_quetzal_analysis_config(None)

    assert config.enabled is True
    assert config.requested_passes == "all"
    assert config.report_path == "/tmp/quetzal-report"


def test_run_quetzal_analysis_writes_report(monkeypatch, tmp_path):
    def fake_extract_export_graph(model, example_inputs, graph_name):
        del model, graph_name
        assert isinstance(example_inputs[0], torch.Tensor)
        assert example_inputs[0].device.type == "cpu"
        return _FakeGraph(["linear", "gelu", "linear"])

    def fake_available_passes():
        return ["fuse_gelu"]

    def fake_run_passes(graph, pass_names):
        assert len(graph.nodes) == 3
        assert pass_names == ["fuse_gelu"]
        return _FakeGraph(["linear", "fused_gelu_linear"]), {"fuse_gelu": {"fused": 1}}

    monkeypatch.setattr(
        quetzal_analysis,
        "_import_quetzal_analysis_tools",
        lambda: (
            fake_extract_export_graph,
            fake_available_passes,
            ["fuse_gelu"],
            fake_run_passes,
        ),
    )

    model = torch.fx.symbolic_trace(
        torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU())
    )
    input_tensor = torch.randn(1, 4)

    quetzal_analysis.run_quetzal_analysis(
        model,
        (input_tensor,),
        {
            quetzal_analysis.QUETZAL_ANALYSIS_PASSES_OPTION: "all",
            quetzal_analysis.QUETZAL_ANALYSIS_REPORT_PATH_OPTION: str(tmp_path),
        },
    )

    reports = list(tmp_path.glob("*.json"))
    assert len(reports) == 1

    report = json.loads(reports[0].read_text())
    assert report["requested_passes"] == "all"
    assert report["executed_passes"] == ["fuse_gelu"]
    assert report["ops_before"] == 3
    assert report["ops_after"] == 2
    assert report["opt_stats"] == {"fuse_gelu": {"fused": 1}}
