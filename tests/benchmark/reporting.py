# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Turning a benchmark run into a standardized, printed, and persisted result.

Owns the result/measurement schema, the human-readable summary printer, the
output-file writer, the TTNN perf-metric aggregation, and the shared
text-generation measurement constructors (used by both the llm and vllm
benchmarks). Pure and device-free.
"""

import json
import os
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional


def get_benchmark_metadata() -> Dict[str, str]:
    """Get common benchmark metadata."""
    return {
        "date": datetime.now().strftime("%d-%m-%Y"),
        "machine_name": socket.gethostname(),
    }


def ttft_measurement(ttft_ms: float) -> Dict[str, Any]:
    """Custom-measurement entry for time-to-first-token (milliseconds).

    Shared vocabulary between the llm and vllm text-generation benchmarks: each
    computes ``ttft_ms`` its own way (on-device iteration timings vs. vLLM
    engine metrics) but agrees here on the measurement shape.
    """
    return {"measurement_name": "ttft", "value": ttft_ms, "target": -1}


def throughput_measurement(samples_per_sec: float) -> Dict[str, Any]:
    """Custom-measurement entry for decode throughput (tokens/samples per second)."""
    return {
        "measurement_name": "samples_per_sec",
        "value": samples_per_sec,
        "target": -1,
    }


def aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results):
    """
    Aggregate TTNN performance metrics from multiple graph files and update results.

    Parameters:
    ----------
    ttnn_perf_metrics_output_file: str
        Base name for the perf metrics files to aggregate.
    results: dict
        Results dictionary to update with aggregated metrics. Modified in place.
    """
    # If the perf_metrics report files exist, load and aggregate results from all graphs
    base_name = os.path.basename(ttnn_perf_metrics_output_file)
    perf_files = [
        f for f in os.listdir(".") if f.startswith(base_name) and f.endswith(".json")
    ]

    if perf_files:
        # Initialize aggregated metrics
        total_ops = 0
        total_shardable_ops = 0
        effectively_sharded_ops = 0
        system_memory_ops = 0
        num_graphs_with_metrics = 0

        for perf_file in sorted(perf_files):
            with open(perf_file, "r") as f:
                perf_metrics_data = json.load(f)

            if "summary" in perf_metrics_data and isinstance(
                perf_metrics_data["summary"], dict
            ):
                summary = perf_metrics_data["summary"]
                total_ops += summary.get("total_ops", 0)
                total_shardable_ops += summary.get("total_shardable_ops", 0)
                effectively_sharded_ops += summary.get("effectively_sharded_ops", 0)
                system_memory_ops += summary.get("system_memory_ops", 0)
                num_graphs_with_metrics += 1

        if num_graphs_with_metrics > 0:
            results["config"]["ttnn_total_ops"] = total_ops
            results["config"]["ttnn_total_shardable_ops"] = total_shardable_ops
            results["config"]["ttnn_effectively_sharded_ops"] = effectively_sharded_ops
            results["config"]["ttnn_system_memory_ops"] = system_memory_ops

            # Calculate aggregated percentage
            if total_shardable_ops > 0:
                results["config"]["ttnn_effectively_sharded_percentage"] = (
                    effectively_sharded_ops / total_shardable_ops
                ) * 100
            else:
                results["config"]["ttnn_effectively_sharded_percentage"] = 0.0

            results["config"]["ttnn_num_graphs"] = num_graphs_with_metrics


def print_benchmark_results(
    model_title: str,
    full_model_name: str,
    model_type: str,
    dataset_name: str,
    date: str,
    machine_name: str,
    total_time: float,
    total_samples: int,
    samples_per_sec: float,
    cpu_samples_per_sec: Optional[float] = None,
    evaluation_score: Optional[float] = None,
    ttft_ms: Optional[float] = None,
    batch_size: int = None,
    data_format: str = None,
    input_size: tuple = None,
    input_sequence_length: Optional[int] = None,
    top1_accuracy: Optional[float] = None,
    top5_accuracy: Optional[float] = None,
    pcc_value: Optional[float] = None,
) -> None:
    """Print formatted benchmark results."""
    print("====================================================================")
    print(f"| {model_title} Benchmark Results:".ljust(67) + "|")
    print("--------------------------------------------------------------------")
    print(f"| Model: {full_model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total samples: {total_samples}")
    print(f"| Avg. decode time: {total_time}")
    print(f"| Avg. samples per second: {samples_per_sec}")

    if cpu_samples_per_sec is not None:
        print(f"| CPU samples per second: {cpu_samples_per_sec}")

    if evaluation_score is not None:
        print(f"| Evaluation score: {evaluation_score}")

    if ttft_ms is not None:
        print(f"| TTFT (ms): {ttft_ms}")

    if batch_size is not None:
        print(f"| Batch size: {batch_size}")

    if data_format is not None:
        print(f"| Data format: {data_format}")

    if input_size is not None:
        print(f"| Input size: {input_size}")

    if input_sequence_length is not None:
        print(f"| Input sequence length: {input_sequence_length}")

    print("====================================================================")

    # Print validation results (Token Accuracy or PCC)
    if top1_accuracy is not None and top5_accuracy is not None:
        print(f"\n=== Token Accuracy Results ===")
        print(f"TOP1 Accuracy: {top1_accuracy * 100:.2f}%")
        print(f"TOP5 Accuracy: {top5_accuracy * 100:.2f}%")
    elif pcc_value is not None:
        print(f"PCC verification passed with PCC={pcc_value:.6f}")


def create_measurement(
    measurement_name: str,
    value: Any,
    step_name: str,
    iteration: int = 1,
    step_warm_up_num_iterations: int = 0,
    target: float = -1,
    device_power: float = -1.0,
    device_temperature: float = -1.0,
) -> Dict[str, Any]:
    """Create a single measurement dictionary."""
    return {
        "iteration": iteration,
        "step_name": step_name,
        "step_warm_up_num_iterations": step_warm_up_num_iterations,
        "measurement_name": measurement_name,
        "value": value,
        "target": target,
        "device_power": device_power,
        "device_temperature": device_temperature,
    }


def create_benchmark_result(
    full_model_name: str,
    model_type: str,
    dataset_name: str,
    num_layers: int,
    batch_size: int,
    input_size: tuple,
    loop_count: int,
    data_format: str,
    total_time: float,
    total_samples: int,
    evaluation_score: Optional[float] = None,
    custom_measurements: Optional[List[Dict[str, Any]]] = None,
    optimization_level: int = 0,
    program_cache_enabled: bool = False,
    trace_enabled: bool = False,
    experimental_weight_dtype: str = "",
    model_info: str = "",
    display_name: str = "",
    torch_xla_enabled: bool = True,
    backend: str = "tt",
    device_name: str = "",
    arch: str = "",
    input_is_image: bool = True,
    input_sequence_length: Optional[int] = -1,
    device_count: int = 1,
    mesh_shape: Optional[tuple] = None,
    vllm: bool = False,
) -> Dict[str, Any]:
    """Create a standardized benchmark result dictionary.

    Args:
        custom_measurements: List of additional measurement dictionaries to include.
                           Each measurement should have keys: measurement_name, value, and optionally
                           iteration, step_name, step_warm_up_num_iterations, target, device_power, device_temperature
    """
    # Create standard measurements
    measurements = [
        create_measurement("total_samples", total_samples, full_model_name),
        create_measurement("total_time", total_time, full_model_name),
    ]

    # Add evaluation score if provided
    if evaluation_score is not None:
        measurements.append(
            create_measurement("evaluation_score", evaluation_score, full_model_name)
        )

    # Add custom measurements if provided
    if custom_measurements:
        for custom_measurement in custom_measurements:
            # Ensure required fields are present
            if (
                "measurement_name" not in custom_measurement
                or "value" not in custom_measurement
            ):
                raise ValueError(
                    "Custom measurements must include 'measurement_name' and 'value' fields"
                )

            # Fill in default values for missing fields
            measurement = {
                "iteration": custom_measurement.get("iteration", 1),
                "step_name": custom_measurement.get("step_name", full_model_name),
                "step_warm_up_num_iterations": custom_measurement.get(
                    "step_warm_up_num_iterations", 0
                ),
                "measurement_name": custom_measurement["measurement_name"],
                "value": custom_measurement["value"],
                "target": custom_measurement.get("target", -1),
                "device_power": custom_measurement.get("device_power", -1.0),
                "device_temperature": custom_measurement.get(
                    "device_temperature", -1.0
                ),
            }
            measurements.append(measurement)

    config = {
        "model_size": "small",
        "optimization_level": optimization_level,
        "program_cache_enabled": program_cache_enabled,
        "trace_enabled": trace_enabled,
        "experimental_weight_dtype": experimental_weight_dtype,
        "model_info": model_info,
        "display_name": display_name,
    }

    if torch_xla_enabled:
        config.update(
            {
                "torch_xla_enabled": torch_xla_enabled,
                "backend": backend,
            }
        )

    image_dimension = ""
    if input_is_image:
        # input_size is (channels, height, width)
        image_dimension = f"{input_size[0]}x{input_size[1]}x{input_size[2]}"

    run_type = f"{'_'.join(full_model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{num_layers}_{loop_count}"
    if vllm:
        run_type += "_vllm"

    return {
        "model": full_model_name,
        "model_type": model_type,
        "run_type": run_type,
        "config": config,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": data_format,
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": input_sequence_length,
        "output_sequence_length": -1,
        "image_dimension": image_dimension,
        "perf_analysis": False,
        "measurements": measurements,
        "device_info": {
            "device_name": device_name,
            "arch": arch,
            "device_count": device_count,
            "mesh_shape": mesh_shape,
            "device_type": None,
        },
    }


def write_benchmark_json(
    results: Dict[str, Any],
    output_file: str,
    *,
    model_rawname: str,
    project: str = "tt-forge/tt-xla",
) -> None:
    """Stamp the dashboard fields onto a result dict and write it as JSON.

    The ``project`` / ``model_rawname`` stamping plus the ``json.dump(indent=2)``
    is identical across every benchmark driver's output-file path, so it lives
    here. Domain-specific post-processing (e.g. the LLM decode-graph perf
    aggregation) should mutate ``results`` *before* calling this.
    """
    results["project"] = project
    results["model_rawname"] = model_rawname
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
