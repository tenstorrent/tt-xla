# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ttxla_tools.logging import logger

QUETZAL_ANALYSIS_PASSES_OPTION = "tt_quetzal_analysis_passes"
QUETZAL_ANALYSIS_REPORT_PATH_OPTION = "tt_quetzal_analysis_report_path"

QUETZAL_ANALYSIS_PASSES_ENV = "TT_TORCH_QUETZAL_ANALYSIS_PASSES"
QUETZAL_ANALYSIS_REPORT_PATH_ENV = "TT_TORCH_QUETZAL_ANALYSIS_REPORT_PATH"


@dataclass(frozen=True)
class QuetzalAnalysisConfig:
    enabled: bool
    requested_passes: str
    report_path: str | None


def get_quetzal_analysis_config(
    options: dict[str, Any] | None,
) -> QuetzalAnalysisConfig:
    requested_passes = _read_option(
        options,
        QUETZAL_ANALYSIS_PASSES_OPTION,
        QUETZAL_ANALYSIS_PASSES_ENV,
    )
    report_path = _read_option(
        options,
        QUETZAL_ANALYSIS_REPORT_PATH_OPTION,
        QUETZAL_ANALYSIS_REPORT_PATH_ENV,
    )

    enabled = bool(requested_passes) and requested_passes.lower() != "none"

    return QuetzalAnalysisConfig(
        enabled=enabled,
        requested_passes=requested_passes or "",
        report_path=report_path,
    )


def run_quetzal_analysis(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[Any, ...],
    options: dict[str, Any] | None,
) -> None:
    config = get_quetzal_analysis_config(options)
    if not config.enabled:
        return

    try:
        extract_export_graph, available_passes, default_passes, run_passes = (
            _import_quetzal_analysis_tools()
        )
        pass_names = _resolve_pass_names(config.requested_passes, default_passes)
        available = set(available_passes())
        missing = [name for name in pass_names if name not in available]
        if missing:
            logger.warning(
                f"[QuetzalAnalysis] Ignoring unknown pass(es): {', '.join(missing)}"
            )
        pass_names = [name for name in pass_names if name in available]
        if not pass_names:
            logger.warning(
                "[QuetzalAnalysis] No valid passes selected after filtering; skipping analysis"
            )
            return

        analysis_module = copy.deepcopy(gm).to("cpu")
        cpu_inputs = tuple(_move_to_cpu(arg) for arg in example_inputs)

        graph_name = getattr(gm, "_get_name", lambda: gm.__class__.__name__)()
        graph = extract_export_graph(analysis_module, cpu_inputs, graph_name=graph_name)
        before = _graph_summary(graph)
        graph_after, opt_stats = run_passes(graph, pass_names)
        after = _graph_summary(graph_after)

        report = {
            "graph_name": graph_name,
            "requested_passes": config.requested_passes,
            "executed_passes": pass_names,
            "ops_before": before["num_ops"],
            "ops_after": after["num_ops"],
            "edges_before": before["num_edges"],
            "edges_after": after["num_edges"],
            "top_ops_before": before["top_ops"],
            "top_ops_after": after["top_ops"],
            "opt_stats": opt_stats,
        }

        logger.info(
            "[QuetzalAnalysis] "
            f"passes={','.join(pass_names)} "
            f"ops={before['num_ops']}->{after['num_ops']} "
            f"edges={before['num_edges']}->{after['num_edges']}"
        )

        if config.report_path:
            report_file = _resolve_report_file(config.report_path, graph_name)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(report, indent=2))
            logger.info(f"[QuetzalAnalysis] Wrote report to {report_file}")
    except Exception as exc:
        logger.warning(f"[QuetzalAnalysis] Failed: {exc}")


def _read_option(
    options: dict[str, Any] | None,
    option_name: str,
    env_name: str,
) -> str | None:
    if options and option_name in options and options[option_name] is not None:
        value = options[option_name]
        return str(value).strip()

    value = os.environ.get(env_name)
    if value is None:
        return None
    return value.strip()


def _import_quetzal_analysis_tools():
    repo_root = Path(__file__).resolve().parents[3]
    sibling_repo = repo_root.parent / "tt-quetzalcoatlus"
    if sibling_repo.is_dir():
        sibling_repo_str = str(sibling_repo)
        if sibling_repo_str not in sys.path:
            sys.path.insert(0, sibling_repo_str)

    from tt_quetzalcoatlus.graph.extract_export import extract_export_graph
    from tt_quetzalcoatlus.graph.transforms import (
        DEFAULT_PASSES,
        available_passes,
        run_passes,
    )

    return extract_export_graph, available_passes, DEFAULT_PASSES, run_passes


def _resolve_pass_names(requested_passes: str, default_passes: list[str]) -> list[str]:
    if requested_passes.lower() == "all":
        return list(default_passes)

    return [name.strip() for name in requested_passes.split(",") if name.strip()]


def _move_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(element) for element in value)
    if isinstance(value, list):
        return [_move_to_cpu(element) for element in value]
    if isinstance(value, dict):
        return {key: _move_to_cpu(element) for key, element in value.items()}
    return value


def _graph_summary(graph) -> dict[str, Any]:
    op_histogram: dict[str, int] = {}
    for node in graph.nodes.values():
        op_histogram[node.op_type] = op_histogram.get(node.op_type, 0) + 1

    top_ops = sorted(op_histogram.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "num_ops": len(graph.nodes),
        "num_edges": len(graph.edges),
        "top_ops": dict(top_ops),
    }


def _resolve_report_file(report_path: str, graph_name: str) -> Path:
    path = Path(report_path)
    if path.suffix == ".json":
        return path

    timestamp_ms = int(time.time() * 1000)
    safe_graph_name = graph_name.replace("/", "_").replace(" ", "_")
    return path / f"{safe_graph_name}_{timestamp_ms}.json"
