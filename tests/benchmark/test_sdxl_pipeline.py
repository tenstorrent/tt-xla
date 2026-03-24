# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from benchmarks.sdxl_pipeline_benchmark import (
    DEFAULT_NUM_INFERENCE_STEPS,
    benchmark_sdxl_pipeline,
)
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

# Defaults
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_LOOP_COUNT = 3
DEFAULT_DATA_FORMAT = "bfloat16"


def _run_sdxl_benchmark(
    output_file,
    request,
    resolution,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    loop_count=DEFAULT_LOOP_COUNT,
    data_format=DEFAULT_DATA_FORMAT,
):
    model_info_name = f"sdxl_pipeline_{resolution}x{resolution}"
    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    print(f"Running SDXL pipeline benchmark: {model_info_name}")
    print(f"""Configuration:
    resolution={resolution}
    num_inference_steps={num_inference_steps}
    optimization_level={optimization_level}
    loop_count={loop_count}
    data_format={data_format}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """)

    results = benchmark_sdxl_pipeline(
        model_info_name=model_info_name,
        display_name=resolved_display_name,
        optimization_level=optimization_level,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        loop_count=loop_count,
        data_format=data_format,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.single_device
def test_sdxl_pipeline_512(output_file, request):
    _run_sdxl_benchmark(
        output_file=output_file,
        request=request,
        resolution=512,
    )


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.single_device
def test_sdxl_pipeline_1024(output_file, request):
    _run_sdxl_benchmark(
        output_file=output_file,
        request=request,
        resolution=1024,
    )
