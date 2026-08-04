# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic text-to-speech benchmark harness for torch-xla / TT.

Sibling of ``imagegen_benchmark.py`` and ``ar_imagegen_benchmark.py``, for models
whose output is an audio waveform. Nothing here is specific to any one TTS model:
per-model wiring supplies a ``build_pipeline_fn`` and the pipeline reports its own
stage names, so adding a TTS model means a new ``*_pipeline.py`` plus a test entry,
not a new harness.

One entry point, ``benchmark_tts_torch_xla``, covering a full text -> waveform
pipeline. Headline throughput is the **real-time factor**: generated audio seconds
per wall-clock second. RTF > 1 means the model synthesizes faster than playback,
which is the number that decides whether a TTS model is usable. A TTS model is
typically several compiled graphs chained by CPU orchestration rather than one
traceable forward, so the harness times the chain and breaks each stage out as its
own ``<stage>_s`` measurement on the same result.

Warmup is by *full* generations, following the AR image harness -- the only way to
compile every graph at the shapes the measured pass reuses. A shortened warmup is
not an option for TTS: the vocoder and any latents stage are shaped by the
generated token count, so a shorter warmup would guarantee a recompile in the
middle of the measured pass. ``WARMUP_PASSES`` of them run before the timed one.

Recompiles are a **hard error**, not a warning. A graph compiling inside the timed
pass inflates the result by a plausible-looking margin and reads downstream as an
ordinary perf regression, so every timed region is bracketed by a ``CompileTime``
counter check.

The ``_perf`` contract a pipeline must populate on each generation:

    _perf = {
        "components": {<stage name>: seconds, ...},  # scalar per-stage times
        "steps": [seconds, ...],                     # per audio-token decode step
        "step_metric_name": "audio_token_step",
        "total": seconds,                            # full generate() wall time
        "audio_samples": int,                        # output waveform length
        "text_tokens": int,                          # optional, for reporting
        "compile_curve": [int, ...],                 # optional, cumulative compiles
    }

Stage times are wall time around a forward plus its XLA sync. Whether they include
a weight upload depends on the pipeline: one that evicts a stage back to the host
after its forward pays to re-upload it next pass, and that cost lands in the stage
time. Pipelines are free to keep stages resident instead (as the XTTS-v2 wiring
does for the stages that would otherwise recompile), in which case stage times are
compute only, matching the other harnesses in tests/benchmark.
"""

import socket
import time

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
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

# Full generations run before the measured pass. See the loop in
# benchmark_tts_torch_xla for why one is not enough.
WARMUP_PASSES = 2

# Minimum share of wall time the measured stages must account for. Below this the
# stage timers are not bracketing real device work -- almost always a missing
# sync_device() in a pipeline wrapper, since torch-xla returns from a forward
# before the graph has run.
MIN_STAGE_TIME_FRACTION = 0.3

MODEL_TYPE = "Audio Generation, Text-to-Speech"
DATASET_NAME = "Text Prompt"


def compile_count() -> int:
    """Cumulative number of graph compilations observed so far."""
    data = met.metric_data("CompileTime")
    return data[0] if data else 0


def sync_device() -> None:
    """Force any pending XLA graph to execute, and wait for it.

    torch-xla is lazy: a forward returns an unevaluated tensor, so a timer closed
    when the call returns measures dynamo dispatch and tracing rather than device
    work. ``sync()`` launches the accumulated graph; ``wait_device_ops()`` blocks
    until the queue drains. ``wait_device_ops()`` on its own is a no-op when
    nothing has been launched, so both are needed.
    """
    torch_xla.sync()
    xm.wait_device_ops()


def assert_no_recompiles(before: int, after: int, what: str) -> None:
    """Fail if any graph compiled between the two counter reads.

    Always prints the delta, so a passing nightly still records the compile counts
    rather than only reporting them on failure.
    """
    print(f"| Compilations during {what}: {after - before} ({before} -> {after})")
    assert after == before, (
        f"{after - before} graph(s) compiled during {what}; the reported timings "
        f"include compilation and are not steady-state performance. Cumulative "
        f"CompileTime count went {before} -> {after}."
    )


def assert_decode_graph_reused(compile_curve) -> None:
    """Fail if the decode loop kept compiling instead of reusing one graph.

    The aggregate before/after check cannot see a decode graph that recompiles
    every token when pass 1 absorbed all of it. The per-step cumulative compile
    curve can: once the loop has warmed up, the curve must be flat.
    """
    if not compile_curve or len(compile_curve) < 4:
        return
    settled_from = len(compile_curve) // 2
    tail_growth = compile_curve[-1] - compile_curve[settled_from]
    assert tail_growth == 0, (
        f"the decode loop kept compiling instead of reusing its graph: "
        f"{tail_growth} new compilation(s) over the last "
        f"{len(compile_curve) - settled_from} of {len(compile_curve)} steps; "
        f"cumulative compile counts per step={compile_curve}"
    )


def assert_stage_timings_plausible(stages_total: float, total: float) -> None:
    """Sanity-check that the stage timers actually bracketed device work."""
    assert stages_total <= total + 1e-6, (
        f"measured stages ({stages_total:.3f}s) exceed the full generation "
        f"({total:.3f}s); stage timers are overlapping or double-counting"
    )
    assert stages_total >= MIN_STAGE_TIME_FRACTION * total, (
        f"measured stages account for only {stages_total / total:.1%} of the "
        f"{total:.3f}s generation; the stage timers are almost certainly closing "
        f"before the device has run (missing sync_device() in a pipeline wrapper)"
    )


def benchmark_tts_torch_xla(
    build_pipeline_fn,
    model_info_name,
    text,
    max_audio_tokens,
    sample_rate,
    optimization_level,
    trace_enabled,
    ttnn_perf_metrics_output_file,
    display_name=None,
    output_wav_path=None,
    save_wav_fn=None,
    data_format="float32",
):
    """Benchmark a text-to-speech pipeline on the TT backend.

    Args:
        build_pipeline_fn: ``build_pipeline_fn(compile_options) -> (pipeline, generate_fn)``.
            ``compile_options`` is forwarded so the pipeline can merge instead of
            overwrite if it needs to switch an option inline.
            ``generate_fn(text, max_audio_tokens) -> waveform tensor`` runs one full
            synthesis and populates ``pipeline._perf``. The pipeline must also
            accept a ``force_num_tokens`` attribute (see below).
        model_info_name: Model name for identification and reporting.
        text: Text to synthesize.
        max_audio_tokens: Cap on generated audio tokens (fixes the work per run so
            the reported RTF is comparable across nightlies).
        sample_rate: Output waveform sample rate in Hz.
        optimization_level: tt-mlir optimization level for compilation.
        trace_enabled: Whether to enable tracing.
        ttnn_perf_metrics_output_file: Base path for TTNN perf metrics files.
        display_name: Display name used for export naming / dashboard.
        output_wav_path: If set, the steady-state waveform is saved here.
        save_wav_fn: ``save_wav_fn(waveform, path)``. Injected rather than imported
            so this module does not depend on any model's audio stack.
        data_format: Precision string for reporting.

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

    # Build the pipeline (registers the "tt" backend; kernels compile lazily on the
    # first forward, i.e. during the warmup pass below).
    pipeline, generate_fn = build_pipeline_fn(options)

    # Warmup: full syntheses that compile every graph at the exact shapes the
    # measured pass reuses. Two passes, not one -- measured on XTTS-v2: the first
    # compiles the graphs, the second settles the configuration once weights are
    # resident (a handful of extra, cheap graphs), and from the third onward the
    # compile count is flat. One warmup pass leaves those stragglers inside the
    # measured pass, which the recompile guard below would (correctly) fail on.
    warmup_tokens = 0
    for i in range(WARMUP_PASSES):
        print(f"Starting warmup pass {i + 1}/{WARMUP_PASSES} (includes compile)...")
        warmup_start = time.perf_counter()
        generate_fn(text, max_audio_tokens)
        warmup_time = time.perf_counter() - warmup_start
        warmup_tokens = len(pipeline._perf["steps"])
        print(
            f"Warmup pass {i + 1}/{WARMUP_PASSES}: {warmup_time:.3f}s "
            f"({warmup_tokens} audio tokens, cumulative compiles={compile_count()})"
        )
        assert warmup_tokens > 0, f"warmup pass {i + 1} generated no audio tokens"

        # Pin every later pass to the token count the first one produced. Sampling
        # is seeded so they agree anyway; pinning makes the shape guarantee
        # explicit rather than incidental, which is what keeps the
        # token-count-shaped stages (vocoder, latents) off the recompile path.
        if pipeline.force_num_tokens is None:
            pipeline.force_num_tokens = warmup_tokens

    # Measured pass (steady-state): every forward must be a cache hit. This is the
    # pass whose audio is saved and whose timing is reported.
    print("Starting steady-state pass...")
    compiles_before = compile_count()
    steady_state_start = time.perf_counter()
    wav = generate_fn(text, max_audio_tokens)
    steady_state_time = time.perf_counter() - steady_state_start
    compiles_after = compile_count()
    print(f"Steady-state pass: {steady_state_time:.3f}s")

    assert_no_recompiles(compiles_before, compiles_after, "the steady-state pass")

    perf = pipeline._perf
    assert_decode_graph_reused(perf.get("compile_curve"))

    if output_wav_path is not None and save_wav_fn is not None:
        save_wav_fn(wav, output_wav_path)
        print(f"Saved output audio to {output_wav_path}")

    components = perf["components"]
    steps = perf["steps"]
    step_metric_name = perf["step_metric_name"]

    steps_total_s = sum(steps)
    stages_total_s = sum(components.values()) + steps_total_s
    assert_stage_timings_plausible(stages_total_s, perf["total"])

    audio_seconds = perf["audio_samples"] / sample_rate
    # "A sample" for TTS is one second of generated audio, so the throughput the
    # dashboard derives as total_samples/total_time is exactly the real-time factor.
    total_samples = audio_seconds
    real_time_factor = audio_seconds / steady_state_time

    step_mean_s = steps_total_s / len(steps) if steps else 0.0
    decode_tokens_per_second = len(steps) / steps_total_s if steps_total_s > 0 else 0.0
    cpu_overhead = max(0.0, perf["total"] - stages_total_s)

    metadata = get_benchmark_metadata()
    full_model_name = model_info_name
    input_sequence_length = perf.get("text_tokens", -1)
    input_size = (input_sequence_length,)

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=MODEL_TYPE,
        dataset_name=DATASET_NAME,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=steady_state_time,
        total_samples=total_samples,
        samples_per_sec=real_time_factor,
        evaluation_score=0.0,
        batch_size=1,
        data_format=data_format,
        input_size=input_size,
        input_sequence_length=input_sequence_length,
    )
    component_lines = "".join(
        f"|   {name} (s):  {value:.3f}\n" for name, value in components.items()
    )
    print(
        f"| Audio tokens: {len(steps)} ({audio_seconds:.2f}s @ {sample_rate} Hz)\n"
        f"| Steady-state:\n"
        f"{component_lines}"
        f"|   {step_metric_name} mean (s):  {step_mean_s:.4f}\n"
        f"|   decode tokens/s:              {decode_tokens_per_second:.2f}\n"
        f"|   CPU overhead (s):             {cpu_overhead:.3f}\n"
        f"|   real-time factor:             {real_time_factor:.3f}x"
    )

    custom_measurements = [
        {"measurement_name": "real_time_factor", "value": real_time_factor},
        {"measurement_name": "e2e_latency", "value": steady_state_time},
        {"measurement_name": f"{step_metric_name}_mean_s", "value": step_mean_s},
        {
            "measurement_name": "decode_tokens_per_second",
            "value": decode_tokens_per_second,
        },
        {"measurement_name": "cpu_overhead_s", "value": cpu_overhead},
    ]
    # One measurement per pipeline stage (e.g. hifigan_s), named by the pipeline.
    for name, value in components.items():
        custom_measurements.append({"measurement_name": f"{name}_s", "value": value})

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=MODEL_TYPE,
        dataset_name=DATASET_NAME,
        num_layers=-1,
        batch_size=1,
        input_size=input_size,
        loop_count=len(steps),
        data_format=data_format,
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
        input_is_image=False,
        input_sequence_length=input_sequence_length,
    )

    return result
