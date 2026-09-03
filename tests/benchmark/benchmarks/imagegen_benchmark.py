# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic text-to-image (diffusion) benchmark harness for torch-xla / TT.

Mirrors the structure of ``vision_benchmark.py``: the per-model configuration
lives in ``test_imagegen.py`` and this module owns the reusable measurement
logic. Diffusion pipelines don't fit the single-forward vision harness — each
generation is a multi-step denoising loop — so this harness offers two schemes,
selected per model by ``staged_residency``:

  - ``staged_residency=False`` (default, resident pipelines): the classic
    two-pass scheme. Pass 1 is a single-step ``generate()`` that triggers the
    first-forward compile of every component; pass 2 is a full
    ``generate(num_inference_steps)`` in which every forward is a cache hit,
    because the components stay on device between the two calls. Pass 2's image
    is saved and its latency drives the reported throughput.

  - ``staged_residency=True``: ONE ``generate()`` call, no outer warmup. A staged
    pipeline loads, runs and frees each component in turn, and eviction discards
    that component's compiled graph along with its weights — the executable pins
    the weight buffers, so weights cannot be freed while the graph is kept. A
    second ``generate()`` therefore recompiles everything, and timing it would
    report cold cycles as warm. Warm cost for these models comes from repeating
    each forward INSIDE its own residency, which the pipeline measures itself and
    reports through ``_perf["cold"]`` / ``_perf["warm"]``.

Per-model wiring provides a ``build_pipeline_fn`` that returns
``(pipeline, generate_fn)``; this module sets the XLA compile options,
builds the pipeline (which compiles the heavy net for TT), runs the measured
pass(es) and emits a standardized benchmark result.
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
    save_image,
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
    staged_residency=False,
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
        staged_residency: True for pipelines that free each component (and its
            compiled graph) before placing the next. Skips the outer warmup pass,
            which for these models would measure recompilation rather than warm
            performance. See the module docstring.

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
    # the first forward, i.e. during the warmup pass below).
    pipeline, generate_fn = build_pipeline_fn(options)

    if staged_residency:
        # No outer warmup: every component is evicted with its graph, so a second
        # generate() would recompile from scratch. Warm cost is measured per
        # component inside its residency by the pipeline itself.
        print("Staged residency: single measured pass (no outer warmup)...")
    else:
        # Pass 1 (warmup): 1 step is enough to trigger the first-forward compile
        # of every component, which stay resident for pass 2.
        print("Starting warmup pass (includes compile)...")
        warmup_start = time.perf_counter()
        generate_fn(prompt, 1)
        warmup_time = time.perf_counter() - warmup_start
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

    # Throughput is reported on the steady-state pass. One image per run.
    total_samples = 1
    samples_per_sec = total_samples / steady_state_time

    # Model-agnostic schema from the pipeline's own instrumentation:
    #   components {name: s}, steps [s], step_metric_name, total s
    perf = pipeline._perf
    components = perf["components"]
    steps = perf["steps"]
    # Staged pipelines also publish per-component cold/warm splits and per-stage
    # compile counters; absent for resident pipelines.
    cold = perf.get("cold") or {}
    warm = perf.get("warm") or {}
    counters = perf.get("counters") or {}
    step_metric_name = perf["step_metric_name"]
    step_mean_s = sum(steps) / len(steps) if steps else 0.0
    tt_components_total = sum(components.values()) + sum(steps)
    cpu_overhead = max(0.0, perf["total"] - tt_components_total)

    # The measured pass recomposed with each component's warm figure: what a
    # resident pipeline gets from its second call, which staged residency cannot
    # run. Emitted only by pipelines publishing warm; e2e_latency is unchanged.
    warm_e2e_latency = None
    if warm:
        warm_step_s = warm.get(step_metric_name, step_mean_s)
        warm_e2e_latency = (
            sum(warm.get(name, value) for name, value in components.items())
            + warm_step_s * len(steps)
            + cpu_overhead
        )

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
    component_lines = "".join(
        f"|   {name} (s):  {value:.3f}\n" for name, value in components.items()
    )
    warm_lines = "".join(
        f"|   {name} warm (s):  {value:.3f}"
        + (f"   (cold {cold[name]:.3f})\n" if name in cold else "\n")
        for name, value in warm.items()
    )
    counter_lines = "".join(
        f"|   {name}: compile {c.get('compile_s', 0.0):.3f}s, "
        f"{c.get('graphs_compiled', 0)} graph(s), wall {c.get('wall_s', 0.0):.3f}s\n"
        for name, c in counters.items()
        if isinstance(c, dict)
    )
    print(
        f"| Num inference steps: {num_inference_steps}\n"
        f"| Steady-state:\n"
        f"{component_lines}"
        f"|   {step_metric_name} mean (s):  {step_mean_s:.3f}\n"
        f"|   CPU overhead (s):    {cpu_overhead:.3f}"
    )
    if warm_lines:
        print(f"| Warm (measured inside residency):\n{warm_lines}", end="")
    if warm_e2e_latency is not None:
        print(f"|   warm-equivalent e2e (s):  {warm_e2e_latency:.3f}")
    if counter_lines:
        print(f"| Compile counters:\n{counter_lines}", end="")

    custom_measurements = [
        {"measurement_name": "images_per_second", "value": samples_per_sec},
        {"measurement_name": "e2e_latency", "value": steady_state_time},
        {"measurement_name": f"{step_metric_name}_mean_s", "value": step_mean_s},
        {"measurement_name": "cpu_overhead_s", "value": cpu_overhead},
    ]
    # One measurement per scalar component (e.g. text_encoder_1_s, vae_s).
    for name, value in components.items():
        custom_measurements.append({"measurement_name": f"{name}_s", "value": value})
    # Warm/cold splits and compile cost, so a regression in recompilation is
    # visible on the dashboard instead of hiding inside the component total.
    for name, value in warm.items():
        custom_measurements.append(
            {"measurement_name": f"{name}_warm_s", "value": value}
        )
    for name, value in cold.items():
        custom_measurements.append(
            {"measurement_name": f"{name}_cold_s", "value": value}
        )
    if warm_e2e_latency is not None:
        custom_measurements.append(
            {"measurement_name": "warm_e2e_latency_s", "value": warm_e2e_latency}
        )
    for name, c in counters.items():
        if not isinstance(c, dict):
            continue
        custom_measurements.append(
            {"measurement_name": f"{name}_compile_s", "value": c.get("compile_s", 0.0)}
        )
        custom_measurements.append(
            {
                "measurement_name": f"{name}_graphs_compiled",
                "value": c.get("graphs_compiled", 0),
            }
        )

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
