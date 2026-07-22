# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic text-to-video (diffusion) pipeline benchmark harness for torch-xla / TT.

Per-model config lives in ``test_video_gen.py``; the reusable measurement logic
lives here. Distinct from ``video_gen_benchmark.py``, which benchmarks single
components (a compiled wrapper + input tensors); this harness drives a full
multi-step generation pipeline and reads its per-stage/per-step instrumentation.

Two-pass scheme:
  - Pass 1 (warmup): a single-step ``generate()`` — triggers the first-forward
    compile of the model's TT components.
  - Pass 2 (steady-state): a full ``generate(num_inference_steps)`` — every
    forward is a cache hit; this pass's video is saved and drives the report.

Per-model wiring provides ``build_pipeline_fn(compile_options) ->
(pipeline, generate_fn)``; this module sets the XLA compile options, builds the
pipeline, runs the two passes, and emits a standardized benchmark result.
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
    save_video,
)

xr.set_device_type("TT")

MODULE_EXPORT_PATH = "modules"


def benchmark_video_gen_pipeline_torch_xla(
    build_pipeline_fn,
    model_info_name,
    prompt,
    num_inference_steps,
    height,
    width,
    num_frames,
    fps,
    optimization_level,
    trace_enabled,
    ttnn_perf_metrics_output_file,
    display_name=None,
    output_video_path=None,
):
    """Benchmark a text-to-video diffusion pipeline on the TT backend.

    Args:
        build_pipeline_fn: ``build_pipeline_fn(compile_options) -> (pipeline, generate_fn)``.
            ``generate_fn(prompt, num_inference_steps) -> list of frames`` runs one
            full text-to-video generation; ``pipeline`` must expose ``_perf`` (see
            below) after each call.
        model_info_name: Model name for identification and reporting.
        prompt: Text prompt to generate from.
        num_inference_steps: Number of denoising steps for the steady-state pass.
        height, width: Output frame dimensions.
        num_frames: Number of frames per generated video.
        fps: Frame rate used when saving the steady-state video.
        optimization_level: tt-mlir optimization level for compilation.
        trace_enabled: Whether to enable tracing.
        ttnn_perf_metrics_output_file: Base path for TTNN perf metrics files.
        display_name: Display name used for export naming / dashboard.
        output_video_path: If set, the steady-state video is saved here.

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

    # Build first so setup() enables SPMD before options trigger client init
    # (else the mesh isn't configured).
    pipeline, generate_fn = build_pipeline_fn(options)
    torch_xla.set_custom_compile_options(options)

    # Pass 1 (warmup): 1 step is enough to trigger the first-forward compile.
    print("Starting warmup pass (includes compile)...")
    warmup_start = time.perf_counter()
    generate_fn(prompt, 1)
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup pass: {warmup_time:.3f}s")

    # Pass 2 (steady-state): the saved video and reported latency come from here.
    print("Starting steady-state pass...")
    steady_state_start = time.perf_counter()
    steady_state_video = generate_fn(prompt, num_inference_steps)
    steady_state_time = time.perf_counter() - steady_state_start
    print(f"Steady-state pass: {steady_state_time:.3f}s")

    if output_video_path is not None:
        save_video(steady_state_video, output_video_path, fps=fps)
        print(f"Saved output video to {output_video_path}")

    # Throughput reported as generated frames per second (num_frames / e2e).
    # Named "generated_..." to distinguish generation speed from the playback
    # fps the mp4 is saved at (they are unrelated).
    total_samples = num_frames
    generated_frames_per_second = total_samples / steady_state_time

    # Per-stage/per-step times from the pipeline's own instrumentation:
    #   _perf = {
    #       "components": {<name>: seconds, ...},   # scalar per-stage times
    #       "steps": [seconds, ...],                # per heavy-net-step times
    #       "step_metric_name": "transformer_step",
    #       "total": seconds,                       # full generate() wall time
    #   }
    perf = pipeline._perf
    components = perf["components"]
    steps = perf["steps"]
    step_metric_name = perf["step_metric_name"]
    step_mean_s = sum(steps) / len(steps) if steps else 0.0
    tt_components_total = sum(components.values()) + sum(steps)
    cpu_overhead = max(0.0, perf["total"] - tt_components_total)

    metadata = get_benchmark_metadata()
    full_model_name = model_info_name
    model_type = "Video Generation, Text-to-Video"
    dataset_name = "Text Prompt"
    input_size = (num_frames, 3, height, width)

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=steady_state_time,
        total_samples=total_samples,
        samples_per_sec=generated_frames_per_second,
        evaluation_score=0.0,
        batch_size=1,
        data_format="bfloat16",
        input_size=input_size,
    )
    component_lines = "".join(
        f"|   {name} (s):  {value:.3f}\n" for name, value in components.items()
    )
    print(
        f"| Num inference steps: {num_inference_steps}\n"
        f"| Num frames: {num_frames}\n"
        f"| Steady-state:\n"
        f"{component_lines}"
        f"|   {step_metric_name} mean (s):  {step_mean_s:.3f}\n"
        f"|   CPU overhead (s):    {cpu_overhead:.3f}"
    )

    custom_measurements = [
        {
            "measurement_name": "generated_frames_per_second",
            "value": generated_frames_per_second,
        },
        {"measurement_name": "e2e_latency", "value": steady_state_time},
        {"measurement_name": f"{step_metric_name}_mean_s", "value": step_mean_s},
        {"measurement_name": "cpu_overhead_s", "value": cpu_overhead},
        {"measurement_name": "num_frames", "value": num_frames},
    ]
    # One measurement per scalar component (e.g. text_encode_s, vae_s).
    for name, value in components.items():
        custom_measurements.append({"measurement_name": f"{name}_s", "value": value})

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
        input_is_image=False,
    )

    return result
