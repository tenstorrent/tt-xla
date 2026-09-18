# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Autoregressive text-to-image benchmark harness for torch-xla / TT.

Sibling of ``imagegen_benchmark.py``, but for *autoregressive* image-token
models (e.g. Janus-Pro) rather than diffusion pipelines. The diffusion harness
is hard-wired to per-step denoising metrics (``te1``/``te2``/``unet_steps``/
``vae``); an AR pipeline instead has a prompt prefill, a long token-by-token
decode loop, and a final vision decode, so it needs its own ``_perf`` schema and
reports decode *tokens/second* as the headline throughput.

Two-pass scheme (same idea as the diffusion harness):

  - Pass 1 (warmup): a full ``generate()``, compiling every graph. Supplies the
    cold numbers.
  - Pass 2 (steady-state): every forward is a cache hit, which holds only because
    the components stay resident. Its image is saved and its timing is
    ``e2e_latency``.

Per-model wiring provides ``build_pipeline_fn -> (pipeline, generate_fn)``. The
pipeline populates ``_perf`` on the schema ``utils.staged_perf_measurements``
consumes, so this harness publishes the same metric names as the diffusion and
video ones.
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


def benchmark_ar_imagegen_torch_xla(
    build_pipeline_fn,
    model_info_name,
    prompt,
    num_image_tokens,
    image_size,
    optimization_level,
    trace_enabled,
    ttnn_perf_metrics_output_file,
    display_name=None,
    output_image_path=None,
):
    """Benchmark an autoregressive text-to-image pipeline on the TT backend.

    Args:
        build_pipeline_fn: ``build_pipeline_fn(compile_options) -> (pipeline, generate_fn)``.
            ``compile_options`` is forwarded so the pipeline can merge instead of
            overwrite if it needs to switch an option inline.
            ``generate_fn(prompt, num_image_tokens) -> image tensor (B, 3, H, W)``
            runs one full text-to-image generation and populates ``pipeline._perf``.
        model_info_name: Model name for identification and reporting.
        prompt: Text prompt to generate from.
        num_image_tokens: Number of image tokens generated per image (AR loop length).
        image_size: Output image height/width (square).
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

    # Build the pipeline (registers the "tt" backend; kernels compile lazily on
    # the first forward, i.e. during the warmup pass below).
    pipeline, generate_fn = build_pipeline_fn(options)

    # Pass 1 (warmup): a full generation compiles every graph (prefill, decode,
    # vision decode) at the shapes the steady-state pass reuses.
    print("Starting warmup pass (includes compile)...")
    warmup_start = time.perf_counter()
    generate_fn(prompt, num_image_tokens)
    warmup_time = time.perf_counter() - warmup_start
    # Deep-copied because a pipeline may clear _perf in place.
    warmup_perf = copy.deepcopy(pipeline._perf)
    print(f"Warmup pass: {warmup_time:.3f}s")

    # Pass 2 (steady-state): every forward is a cache hit; this image is saved.
    print("Starting steady-state pass...")
    steady_state_start = time.perf_counter()
    steady_state_image = generate_fn(prompt, num_image_tokens)
    steady_state_time = time.perf_counter() - steady_state_start
    print(f"Steady-state pass: {steady_state_time:.3f}s")

    if output_image_path is not None:
        save_image(steady_state_image, output_image_path)
        print(f"Saved output image to {output_image_path}")

    # One image per run.
    total_samples = 1
    samples_per_sec = total_samples / steady_state_time

    # Per-stage times from the pipeline's own instrumentation (steady-state pass).
    perf = pipeline._perf
    step_metric_name = perf["step_metric_name"]
    derived = staged_perf_measurements(
        perf,
        step_metric=step_metric_name,
        step_name=model_info_name,
        warmup_perf=warmup_perf,
        staged_residency=getattr(pipeline, "benchmark_staged_residency", False),
    )
    # On the warm step, so the build-carrying first step does not drag it down.
    warm_step_s = derived["warm"].get(step_metric_name, 0.0)
    decode_tokens_per_second = 1.0 / warm_step_s if warm_step_s > 0 else 0.0

    metadata = get_benchmark_metadata()
    full_model_name = model_info_name
    model_type = "Image Generation, Text-to-Image"
    dataset_name = "Text Prompt"
    input_size = (3, image_size, image_size)

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
        f"| Num image tokens: {num_image_tokens}\n"
        f"| Per component, warm (cold in brackets):\n"
        f"{format_staged_perf_summary(derived, step_metric_name)}\n"
        f"|   Decode tokens/s:     {decode_tokens_per_second:.2f}"
    )

    custom_measurements = [
        create_measurement("images_per_second", samples_per_sec, full_model_name),
        # Measured wall clock of the steady pass; the comparable pair is
        # e2e_warm_s / e2e_cold_s.
        create_measurement("e2e_latency", steady_state_time, full_model_name),
        create_measurement(
            "decode_tokens_per_second",
            decode_tokens_per_second,
            full_model_name,
            step_warm_up_num_iterations=1,
        ),
    ]
    custom_measurements.extend(derived["measurements"])

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=1,
        input_size=input_size,
        loop_count=num_image_tokens,
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
        input_is_image=True,
    )

    return result
