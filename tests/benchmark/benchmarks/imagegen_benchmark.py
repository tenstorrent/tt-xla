# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic text-to-image (diffusion) benchmark harness for torch-xla / TT.

Mirrors the structure of ``vision_benchmark.py``: the per-model configuration
lives in ``test_imagegen.py`` and this module owns the reusable measurement
logic. Diffusion pipelines don't fit the single-forward vision harness — each
generation is a multi-step denoising loop — so this harness uses a two-pass
scheme:

  - Pass 1 (cold): the first ``generate()`` call; the heavy net (UNet /
    transformer) compiles on its first forward, so this pass includes compile
    time.
  - Pass 2 (warm): a second ``generate()`` call; every forward is a cache hit.
    This is the steady-state pass whose image is saved and whose latency drives
    the reported throughput.

Per-model wiring provides a ``build_pipeline_fn`` that returns
``(pipeline, generate_fn, save_fn)``; this module sets the XLA compile options,
builds the pipeline (which compiles the heavy net for TT), runs the two passes
and emits a standardized benchmark result.
"""

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
        build_pipeline_fn: Callable returning ``(pipeline, generate_fn, save_fn)``.
            ``generate_fn(num_inference_steps) -> image tensor (B, 3, H, W)`` runs
            one full text-to-image generation. ``save_fn(image, path)`` writes the
            decoded image (may be ``None`` to skip saving). The heavy net is
            expected to already be compiled for / moved to TT inside this call.
        model_info_name: Model name for identification and reporting.
        prompt: Text prompt to generate from.
        num_inference_steps: Number of denoising steps per generation.
        height, width: Output image dimensions.
        optimization_level: tt-mlir optimization level for compilation.
        trace_enabled: Whether to enable tracing.
        ttnn_perf_metrics_output_file: Base path for TTNN perf metrics files.
        display_name: Display name used for export naming / dashboard.
        output_image_path: If set, the warm-pass image is saved here.

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

    # Build + compile the pipeline (heavy net registers the "tt" backend and is
    # moved to the XLA device here; actual kernel compilation happens lazily on
    # the first forward, i.e. during the cold pass below).
    pipeline, generate_fn, save_fn = build_pipeline_fn()

    # Pass 1 (cold): first generation, includes first-forward compile.
    print("Starting cold pass (includes compile)...")
    cold_start = time.perf_counter()
    generate_fn(num_inference_steps)
    cold_time = time.perf_counter() - cold_start
    print(f"Cold pass: {cold_time:.3f}s")

    # Pass 2 (warm): steady-state generation; this image is the saved one.
    print("Starting warm pass (steady-state)...")
    warm_start = time.perf_counter()
    warm_image = generate_fn(num_inference_steps)
    warm_time = time.perf_counter() - warm_start
    print(f"Warm pass: {warm_time:.3f}s")

    if output_image_path is not None and save_fn is not None:
        save_fn(warm_image, output_image_path)
        print(f"Saved output image to {output_image_path}")

    # Throughput is reported on the steady-state (warm) pass. One image per run.
    total_samples = 1
    samples_per_sec = total_samples / warm_time
    per_step_latency_ms = (warm_time / num_inference_steps) * 1000.0
    # First-forward compile overhead, approximated as the cold/warm delta.
    compile_time = max(0.0, cold_time - warm_time)

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
        total_time=warm_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        evaluation_score=0.0,
        batch_size=1,
        data_format="bfloat16",
        input_size=input_size,
    )
    print(
        f"| Num inference steps: {num_inference_steps}\n"
        f"| Per-step latency (ms): {per_step_latency_ms:.3f}\n"
        f"| Compile time (cold-warm) (s): {compile_time:.3f}"
    )

    custom_measurements = [
        {"measurement_name": "images_per_second", "value": samples_per_sec},
        {"measurement_name": "e2e_latency", "value": warm_time},
        {"measurement_name": "per_step_latency_ms", "value": per_step_latency_ms},
        {"measurement_name": "compile_time", "value": compile_time},
    ]

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=1,
        input_size=input_size,
        loop_count=num_inference_steps,
        data_format="bfloat16",
        total_time=warm_time,
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
        input_is_image=True,
    )

    return result
