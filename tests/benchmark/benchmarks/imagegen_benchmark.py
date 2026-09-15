# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic text-to-image (diffusion) benchmark harness for torch-xla / TT.

Mirrors ``vision_benchmark.py``: per-model configuration lives in
``test_imagegen.py`` and this module owns the measurement logic. A generation is
a multi-step denoising loop, so it runs one of two schemes, chosen by the
pipeline's own ``benchmark_staged_residency``:

  - resident (the default): a single-step warmup then a full
    ``generate(num_inference_steps)`` whose every forward is a cache hit. The
    second pass gives the image and the warm numbers, the warmup the cold ones.

  - staged: ONE call. Eviction discards a component's compiled graph with its
    weights, so a second call would recompile and report cold cycles as warm.
    These models repeat each forward inside its own residency and report both
    halves through ``_perf["cold"]`` / ``_perf["warm"]``.

Both publish the same metric names for the same quantities -- see
``utils.staged_perf_measurements``, shared with the video-generation harness.
``e2e_latency`` is the exception: the measured wall clock, warm for a resident
model and cold for a staged one, so ``e2e_warm_s`` / ``e2e_cold_s`` are the
comparable pair.

Per-model wiring provides ``build_pipeline_fn -> (pipeline, generate_fn)``.
"""

import copy
import socket
import time

import torch_xla
import torch_xla.runtime as xr
from utils import (
    build_xla_export_name,
    create_benchmark_result,
    create_measurement,
    format_staged_perf_summary,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
    save_image,
    staged_perf_measurements,
)

xr.set_device_type("TT")

MODULE_EXPORT_PATH = "modules"


def benchmark_imagegen_torch_xla(
    build_pipeline_fn,
    model_info_name,
    prompt,
    num_inference_steps,
    height,
    width,
    optimization_level,
    trace_enabled,
    ttnn_perf_metrics_output_file,
    display_name=None,
    output_image_path=None,
):
    """Benchmark a text-to-image diffusion pipeline on the TT backend.

    Args:
        build_pipeline_fn: ``build_pipeline_fn(compile_options) -> (pipeline, generate_fn)``.
            ``compile_options`` is forwarded so the pipeline can merge instead
            of overwriting if it needs to switch any option inline.
            ``generate_fn(prompt, num_inference_steps) -> image tensor (B, 3, H, W)``
            runs one full text-to-image generation.
        model_info_name: Model name for identification and reporting.
        prompt: Text prompt to generate from.
        num_inference_steps: Number of denoising steps per generation.
        height, width: Output image dimensions.
        optimization_level: tt-mlir optimization level for compilation.
        trace_enabled: Whether to enable tracing.
        ttnn_perf_metrics_output_file: Base path for TTNN perf metrics files.
        display_name: Display name used for export naming / dashboard.
        output_image_path: If set, the steady-state image is saved here.

    Returns:
        Standardized benchmark result dict (see ``create_benchmark_result``).
    """
    export_model_name = build_xla_export_name(
        model_name=display_name or model_info_name,
        num_layers=None,
        batch_size=1,
        input_sequence_length=None,
    )

    options = {
        "optimization_level": optimization_level,
        "export_path": MODULE_EXPORT_PATH,
        "export_model_name": export_model_name,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
        "enable_trace": trace_enabled,
    }
    torch_xla.set_custom_compile_options(options)

    # Registers the "tt" backend; kernels compile lazily on the first forward.
    pipeline, generate_fn = build_pipeline_fn(options)

    # Declared by the pipeline, next to the code it describes.
    staged_residency = getattr(pipeline, "benchmark_staged_residency", False)

    warmup_perf = None
    if staged_residency:
        print("Staged residency: single measured pass (no outer warmup)...")
    else:
        # 1 step is enough to compile every component.
        print("Starting warmup pass (includes compile)...")
        warmup_start = time.perf_counter()
        generate_fn(prompt, 1)
        warmup_time = time.perf_counter() - warmup_start
        # Deep-copied because a pipeline may clear _perf in place rather than
        # rebuilding it, which would empty this before it is read.
        warmup_perf = copy.deepcopy(pipeline._perf)
        print(f"Warmup pass: {warmup_time:.3f}s")

    # The measured pass; this image is the saved one.
    print("Starting steady-state pass...")
    steady_state_start = time.perf_counter()
    steady_state_image = generate_fn(prompt, num_inference_steps)
    steady_state_time = time.perf_counter() - steady_state_start
    print(f"Steady-state pass: {steady_state_time:.3f}s")

    if output_image_path is not None:
        save_image(steady_state_image, output_image_path)
        print(f"Saved output image to {output_image_path}")

    # Model-agnostic schema from the pipeline's own instrumentation:
    #   components {name: s}, steps [s], step_metric_name, total s
    perf = pipeline._perf
    step_metric_name = perf["step_metric_name"]
    derived = staged_perf_measurements(
        perf,
        step_metric=step_metric_name,
        step_name=model_info_name,
        warmup_perf=warmup_perf,
        staged_residency=staged_residency,
    )

    # Throughput stays on the wall clock: a staged pipeline recompiles every
    # call, so its e2e_warm_s describes a call it never makes.
    total_samples = 1
    samples_per_sec = total_samples / steady_state_time

    metadata = get_benchmark_metadata()
    full_model_name = model_info_name
    model_type = "Image Generation, Text-to-Image"
    dataset_name = "Text Prompt"
    input_size = (3, height, width)

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=steady_state_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        evaluation_score=0.0,
        batch_size=1,
        data_format="bfloat16",
        input_size=input_size,
    )
    print(
        f"| Num inference steps: {num_inference_steps}\n"
        f"| Per component, warm (cold in brackets):\n"
        f"{format_staged_perf_summary(derived, step_metric_name)}"
    )

    custom_measurements = [
        create_measurement("images_per_second", samples_per_sec, full_model_name),
        create_measurement("e2e_latency", steady_state_time, full_model_name),
    ]
    custom_measurements.extend(derived["measurements"])
    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=1,
        input_size=input_size,
        loop_count=num_inference_steps,
        data_format="bfloat16",
        total_time=steady_state_time,
        total_samples=total_samples,
        evaluation_score=0.0,
        custom_measurements=custom_measurements,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        model_info=model_info_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
        device_count=xr.global_runtime_device_count(),
        mesh_shape=getattr(pipeline, "mesh_shape", None),
        input_is_image=True,
    )

    return result
