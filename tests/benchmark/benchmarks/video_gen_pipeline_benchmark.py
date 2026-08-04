# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end benchmark harness for video-generation *pipelines*.

``video_gen_benchmark.py`` benchmarks one component at a time (a single forward
pass of the DiT, VAE or text encoder). This harness benchmarks the whole
pipeline: one call produces one video, so the reported figure is
generation latency rather than per-forward latency.

Structurally this mirrors ``imagegen_benchmark.benchmark_imagegen_torch_xla``
(warmup pass at 1 step to force compilation, then a steady-state pass at the
real step count) and consumes the same model-agnostic ``pipeline._perf``
schema, so a video pipeline and an image pipeline report comparably.

Reported metrics
----------------
``total_time`` and ``total_samples`` are the primary pair — the Forge dashboard
derives secs/sample from ``total_time / total_samples`` and falls back to
``forward_pass_time_ms``, so both are emitted. Per-stage times (text encoder,
VAE, mean DiT step, CPU overhead) come through as custom measurements.
"""

# Built-in modules
import socket
import time

import torch_xla
import torch_xla.runtime as xr
from utils import (
    build_xla_export_name,
    create_benchmark_result,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
)

xr.set_device_type("TT")

# Where the compiler dumps stablehlo/ttir/ttnn + flatbuffer artifacts; the perf
# CI collects them from here (matches the other harnesses).
MODULE_EXPORT_PATH = "modules"

# Steps used for the warmup generation. One step is enough to trigger the
# first-forward compile of every component, including both DiT experts.
WARMUP_STEPS = 1


def benchmark_video_gen_pipeline_torch_xla(
    *,
    build_pipeline_fn,
    model_info_name,
    display_name,
    prompt,
    num_inference_steps,
    compiler_config,
    ttnn_perf_metrics_output_file,
    frame_shape,
    data_format="bfloat16",
):
    """Benchmark a full video-generation pipeline on the TT backend.

    Args:
        build_pipeline_fn: ``fn(compile_options) -> (pipeline, generate_fn)``.
            ``generate_fn(prompt, num_inference_steps)`` runs one full
            generation; ``pipeline._perf`` carries the per-stage timings.
        model_info_name: Full model name for reporting. For the dashboard's E2E
            view this must contain "E2E" (see tests/benchmark/test_wan.py).
        display_name: Short name used for export / perf-metric file naming.
        num_inference_steps: Denoising steps for the steady-state pass.
        compiler_config: ``CompilerConfig`` supplying tt-mlir compile options.
        frame_shape: ``(channels, height, width)`` of one output frame, for
            reporting.
        data_format: Precision string for reporting.

    Returns:
        Standardized benchmark result dictionary.
    """
    xr.set_device_type("TT")

    options = compiler_config.to_torch_compile_options()
    options["export_path"] = MODULE_EXPORT_PATH
    options["export_model_name"] = build_xla_export_name(
        model_name=display_name,
        num_layers=None,
        batch_size=1,
        input_sequence_length=None,
    )
    options["ttnn_perf_metrics_enabled"] = True
    options["ttnn_perf_metrics_output_file"] = ttnn_perf_metrics_output_file

    # Compile options are set inside the builder too (it must set them before
    # the first XLA op), but set them here as well so a builder that does not
    # is still covered.
    torch_xla.set_custom_compile_options(options)

    pipeline, generate_fn = build_pipeline_fn(options)

    print("Starting warmup pass (includes compile)...")
    warmup_start = time.perf_counter()
    generate_fn(prompt, WARMUP_STEPS)
    print(f"Warmup pass: {time.perf_counter() - warmup_start:.3f}s")

    print("Starting steady-state pass...")
    steady_state_start = time.perf_counter()
    generate_fn(prompt, num_inference_steps)
    steady_state_time = time.perf_counter() - steady_state_start
    print(f"Steady-state pass: {steady_state_time:.3f}s")

    # One video per generation.
    total_samples = 1
    samples_per_sec = total_samples / steady_state_time

    perf = pipeline._perf
    components = perf["components"]
    steps = perf["steps"]
    step_metric_name = perf["step_metric_name"]
    step_mean_s = sum(steps) / len(steps) if steps else 0.0
    tt_components_total = sum(components.values()) + sum(steps)
    cpu_overhead = max(0.0, perf["total"] - tt_components_total)

    metadata = get_benchmark_metadata()
    model_type = "Video Generation, Image-to-Video"
    dataset_name = "Text Prompt + First Frame"

    print_benchmark_results(
        model_title=model_info_name,
        full_model_name=model_info_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=steady_state_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        evaluation_score=None,
        batch_size=1,
        data_format=data_format,
        input_size=frame_shape,
    )
    component_lines = "".join(
        f"|   {name} (s):  {value:.3f}\n" for name, value in components.items()
    )
    print(
        f"| Num inference steps: {num_inference_steps}\n"
        f"| Denoise steps measured: {len(steps)}\n"
        f"| Steady-state:\n"
        f"{component_lines}"
        f"|   {step_metric_name} mean (s):  {step_mean_s:.3f}\n"
        f"|   CPU overhead (s):    {cpu_overhead:.3f}"
    )

    custom_measurements = [
        {"measurement_name": "videos_per_second", "value": samples_per_sec},
        {"measurement_name": "e2e_latency", "value": steady_state_time},
        {"measurement_name": f"{step_metric_name}_mean_s", "value": step_mean_s},
        {"measurement_name": "cpu_overhead_s", "value": cpu_overhead},
        # The dashboard falls back to forward_pass_time_ms when the
        # total_time/total_samples pair is unavailable; for a pipeline the
        # equivalent "one forward" is one full generation.
        {
            "measurement_name": "forward_pass_time_ms",
            "value": steady_state_time * 1000.0,
            "iteration": 1,
        },
    ]
    for name, value in components.items():
        custom_measurements.append({"measurement_name": f"{name}_s", "value": value})
    # One measurement per denoise step, so step-level variance is visible.
    for i, seconds in enumerate(steps):
        custom_measurements.append(
            {
                "measurement_name": f"{step_metric_name}_time_ms",
                "value": seconds * 1000.0,
                "iteration": i + 1,
            }
        )

    return create_benchmark_result(
        full_model_name=model_info_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=1,
        input_size=frame_shape,
        loop_count=num_inference_steps,
        data_format=data_format,
        total_time=steady_state_time,
        total_samples=total_samples,
        evaluation_score=None,
        custom_measurements=custom_measurements,
        optimization_level=compiler_config.optimization_level,
        program_cache_enabled=True,
        trace_enabled=compiler_config.enable_trace,
        model_info=model_info_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
        device_count=xr.global_runtime_device_count(),
        input_is_image=False,
    )
