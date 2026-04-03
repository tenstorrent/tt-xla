# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from benchmarks.sdxl_pipeline_benchmark import SDXLConstants, benchmark_sdxl_pipeline
from utils import aggregate_ttnn_perf_metrics, resolve_display_name


def assert_no_perf_regression(measurements, device_type, resolution):
    key = (device_type, resolution)
    thresholds = SDXLConstants.PERF_THRESHOLDS.get(key)

    if thresholds is None:
        print(f"[perf-regression-check] No baseline for {key}; skipping.")
        return

    failures = []
    for measurement in measurements:
        name = measurement.get("measurement_name", "")
        if name not in SDXLConstants.CHECKED_METRICS or name not in thresholds:
            continue
        value = measurement["value"]
        limit = thresholds[name]
        status = "PASS" if value <= limit else "FAIL"
        print(
            f"[perf-regression-check] {name}: measured={value:.4f}s "
            f"limit={limit:.4f}s"
        )
        if value > limit:
            failures.append(
                f"  {name}: {value:.4f}s exceeds limit {limit:.4f}s "
                f"by {(value - limit) / limit * 100:.1f}%"
            )

    if failures:
        pytest.fail(
            f"Performance regression detected on {device_type} at resolution {resolution}:\n"
            + "\n".join(failures)
        )


def run_sdxl_benchmark(
    output_file,
    request,
    resolution,
    num_inference_steps=SDXLConstants.NUM_INFERENCE_STEPS,
    optimization_level=SDXLConstants.OPTIMIZATION_LEVEL,
    loop_count=SDXLConstants.LOOP_COUNT,
    data_format=SDXLConstants.DATA_FORMAT,
    perf_regression_check=False,
):
    model_info_name = f"sdxl_pipeline_{resolution}x{resolution}"
    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    print(f"Running SDXL pipeline benchmark: {model_info_name}")
    print(
        f"Configuration:\n"
        f"    resolution={resolution}\n"
        f"    num_inference_steps={num_inference_steps}\n"
        f"    optimization_level={optimization_level}\n"
        f"    loop_count={loop_count}\n"
        f"    data_format={data_format}\n"
        f"    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}"
    )

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

    device_type = results["device_info"]["device_type"]
    thresholds = SDXLConstants.PERF_THRESHOLDS.get((device_type, resolution), {})
    for measurement in results["measurements"]:
        name = measurement.get("measurement_name", "")
        if name in thresholds:
            measurement["target"] = thresholds[name]

    if perf_regression_check:
        assert_no_perf_regression(
            measurements=results["measurements"],
            device_type=device_type,
            resolution=resolution,
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
def test_sdxl_pipeline_512(output_file, request, perf_regression_check):
    run_sdxl_benchmark(
        output_file=output_file,
        request=request,
        resolution=512,
        loop_count=1,
        perf_regression_check=perf_regression_check,
    )


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.single_device
def test_sdxl_pipeline_1024(output_file, request, perf_regression_check):
    run_sdxl_benchmark(
        output_file=output_file,
        request=request,
        resolution=1024,
        loop_count=1,
        perf_regression_check=perf_regression_check,
    )
