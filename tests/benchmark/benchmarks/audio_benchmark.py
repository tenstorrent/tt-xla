# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic audio-pipeline benchmark harness for torch-xla / TT.

Sibling of ``imagegen_benchmark.py`` / ``ar_imagegen_benchmark.py``, for models
whose input or output is audio and whose generation is an autoregressive decode
loop: audio-understanding models that answer in text (Voxtral) and TTS models
that emit audio tokens (XTTS-v2, Llasa). The per-model configuration lives in
``test_audio.py``; this module owns the reusable measurement logic.

Two-pass scheme (same idea as the image-gen harnesses):

  - Pass 1 (warmup): a full ``generate()``. It has to use the *same* token
    budget as the measured pass: these pipelines decode over a static
    right-padded window whose shape is ``prefill + max_new_tokens``, so a
    shorter warmup would compile a different shape and leave the compile in the
    measured pass.
  - Pass 2 (steady-state): a second full ``generate()`` where every forward is
    a cache hit. This is the pass that is reported.

The pipeline reports its own per-stage timings in ``pipeline._perf``, in the
schema shared with the image-gen harness — ``components`` and ``steps`` hold
seconds, counts live in their own top-level keys::

    _perf = {
        "components": {<name>: seconds, ...},   # scalar per-stage device times
        "steps": [seconds, ...],                # one entry per decode step
        "step_metric_name": "decode_step",      # e.g. "audio_token_step"
        "total": seconds,                       # full generate() wall time
        <count_name>: int,                      # e.g. prefill_tokens, output_tokens
    }

Every extra scalar key is emitted verbatim as a dashboard measurement, so a
pipeline can report its own counts (``prefill_tokens``, ``audio_samples``, ...)
without touching this harness.

Headline throughput is decode tokens/second, and ``total_time`` /
``total_samples`` are the decode-loop totals — the same convention as
``llm_benchmark.py``, so a generative audio model is comparable with the LLMs on
the dashboard. ``steps[0]`` is reported as TTFT: these pipelines carry no KV
cache, so the first step is the one that also computes the prompt.
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

# Keys of the ``_perf`` schema that this harness interprets itself; any other
# scalar key is passed through to the dashboard as-is.
_PERF_RESERVED_KEYS = frozenset({"components", "steps", "step_metric_name", "total"})


def benchmark_audio_torch_xla(
    build_pipeline_fn,
    model_info_name,
    num_generation_steps,
    optimization_level,
    trace_enabled,
    ttnn_perf_metrics_output_file,
    model_type,
    dataset_name,
    display_name=None,
    save_output_fn=None,
):
    """Benchmark an audio generation/understanding pipeline on the TT backend.

    Args:
        build_pipeline_fn: ``build_pipeline_fn(compile_options) -> (pipeline, generate_fn)``.
            ``compile_options`` is forwarded so the pipeline can merge instead of
            overwrite if it needs to switch an option inline.
            ``generate_fn(num_generation_steps) -> output`` runs one full
            generation and populates ``pipeline._perf``.
        model_info_name: Model name for identification and reporting.
        num_generation_steps: Token budget per generation (the decode-loop cap).
        optimization_level: tt-mlir optimization level for compilation.
        trace_enabled: Whether to enable tracing.
        ttnn_perf_metrics_output_file: Base path for TTNN perf metrics files.
        model_type: Dashboard model type (e.g. "audio-to-text").
        dataset_name: Dashboard dataset name (e.g. "Audio Prompt").
        display_name: Display name used for export naming / dashboard.
        save_output_fn: If set, ``save_output_fn(output)`` persists the
            steady-state output (a text answer, a waveform, ...).

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

    # Build the pipeline (registers the "tt" backend and moves the model to the
    # XLA device; kernels compile lazily on the first forward, i.e. during the
    # warmup pass below).
    pipeline, generate_fn = build_pipeline_fn(options)

    # Pass 1 (warmup): same token budget as the measured pass, so the decode
    # window has the shape the steady-state pass reuses.
    print("Starting warmup pass (includes compile)...")
    warmup_start = time.perf_counter()
    generate_fn(num_generation_steps)
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup pass: {warmup_time:.3f}s")

    # Pass 2 (steady-state): every forward is a cache hit; this is the reported
    # pass and the output that gets saved.
    print("Starting steady-state pass...")
    steady_state_start = time.perf_counter()
    steady_state_output = generate_fn(num_generation_steps)
    steady_state_time = time.perf_counter() - steady_state_start
    print(f"Steady-state pass: {steady_state_time:.3f}s")

    if save_output_fn is not None:
        save_output_fn(steady_state_output)

    # Per-stage times from the pipeline's own instrumentation (steady-state pass).
    perf = pipeline._perf
    components = perf["components"]
    steps = perf["steps"]
    step_metric_name = perf["step_metric_name"]
    if not steps:
        raise AssertionError(
            f"{model_info_name}: the pipeline reported no generation steps"
        )

    # No KV cache in these loops: step 0 is the one that also computes the
    # prompt, so it is TTFT and is kept out of the steady-state decode stats.
    ttft_ms = steps[0] * 1000.0
    decode_steps = steps[1:]
    decode_total_s = sum(decode_steps)
    decode_step_mean_s = decode_total_s / len(decode_steps) if decode_steps else 0.0
    tokens_per_second = (
        len(decode_steps) / decode_total_s if decode_total_s > 0 else 0.0
    )
    tt_stages_total = sum(components.values()) + sum(steps)
    cpu_overhead = max(0.0, perf["total"] - tt_stages_total)

    # Counts (and any other scalar) the pipeline reports next to the timings.
    extra_measurements = {
        name: value
        for name, value in perf.items()
        if name not in _PERF_RESERVED_KEYS and isinstance(value, (int, float))
    }
    prefill_tokens = extra_measurements.get("prefill_tokens")

    metadata = get_benchmark_metadata()
    full_model_name = model_info_name

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=decode_total_s,
        total_samples=len(decode_steps),
        samples_per_sec=tokens_per_second,
        evaluation_score=0.0,
        ttft_ms=ttft_ms,
        batch_size=1,
        data_format="bfloat16",
        input_sequence_length=prefill_tokens,
    )
    component_lines = "".join(
        f"|   {name} (s):  {value:.3f}\n" for name, value in components.items()
    )
    extra_lines = "".join(
        f"|   {name}:  {value}\n" for name, value in extra_measurements.items()
    )
    print(
        f"| Token budget: {num_generation_steps}\n"
        f"| Steady-state:\n"
        f"{component_lines}"
        f"{extra_lines}"
        f"|   TTFT (s):                 {steps[0]:.3f}\n"
        f"|   {step_metric_name} mean (s):  {decode_step_mean_s:.4f}\n"
        f"|   Decode tokens/s:          {tokens_per_second:.2f}\n"
        f"|   e2e latency (s):          {steady_state_time:.3f}\n"
        f"|   CPU overhead (s):         {cpu_overhead:.3f}"
    )

    custom_measurements = [
        {"measurement_name": "tokens_per_second", "value": tokens_per_second},
        {"measurement_name": "ttft", "value": ttft_ms},
        {"measurement_name": "e2e_latency", "value": steady_state_time},
        {"measurement_name": f"{step_metric_name}_mean_s", "value": decode_step_mean_s},
        {"measurement_name": "cpu_overhead_s", "value": cpu_overhead},
    ]
    # One measurement per scalar component (e.g. speaker_encoder_s, hifigan_s).
    for name, value in components.items():
        custom_measurements.append({"measurement_name": f"{name}_s", "value": value})
    # Pipeline-reported counts (prefill_tokens, output_tokens, audio_samples, ...).
    for name, value in extra_measurements.items():
        custom_measurements.append({"measurement_name": name, "value": value})

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=1,
        input_size=(prefill_tokens if prefill_tokens is not None else -1,),
        loop_count=num_generation_steps,
        data_format="bfloat16",
        total_time=decode_total_s,
        total_samples=len(decode_steps),
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
        input_sequence_length=prefill_tokens if prefill_tokens is not None else -1,
    )

    return result
