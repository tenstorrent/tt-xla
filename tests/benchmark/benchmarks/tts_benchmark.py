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
        "compile_curve": [int, ...],                 # cumulative compiles per step
    }

``compile_curve`` is required: it is the only thing that can show a decode loop
recompiling per token, which the aggregate check misses if warmup absorbed it.
Reading it needs torch-xla's counters, so the per-model wiring records it.

Stage times are wall time around a forward plus its XLA sync. Pipelines are
expected to place modules on device once and leave them there, so a stage time is
compute rather than a weight upload, and the numbers stay comparable.
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

    Once the loop has warmed up the cumulative curve must be flat. An absent or
    too-short curve is a wiring bug, not a reason to skip the check.
    """
    assert compile_curve, (
        "the pipeline recorded no _perf['compile_curve'], so the decode loop "
        "cannot be checked for per-token recompilation; the per-model wiring "
        "must append compile_count() after each decode call"
    )
    assert len(compile_curve) >= 4, (
        f"too few decode steps to judge graph reuse: {compile_curve}; the "
        f"benchmark needs a generation long enough for the curve to settle"
    )
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
            synthesis with those settings and populates ``pipeline._perf``. It must
            honour both arguments, and must generate the same number of audio
            tokens every time it is called with the same ones -- see
            ``max_audio_tokens``.
        model_info_name: Model name for identification and reporting.
        text: Text to synthesize.
        max_audio_tokens: Audio tokens to generate per run. The pipeline is
            expected to produce a length that depends only on this value, not on
            what sampling happened to yield, so that the length-shaped stages
            (vocoder, latents) see identical shapes in the warmup and measured
            passes and cannot recompile mid-measurement. Fixing the work per run
            is also what makes the reported RTF comparable across nightlies.
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

    # Warmup: full syntheses at the shapes the measured pass reuses. Two passes,
    # not one -- measured on XTTS-v2, pass 1 compiles the graphs, pass 2 compiles a
    # few more once buffers are in their steady state, pass 3 onward is flat.
    pass_tokens = []
    for i in range(WARMUP_PASSES):
        print(f"Starting warmup pass {i + 1}/{WARMUP_PASSES} (includes compile)...")
        warmup_start = time.perf_counter()
        generate_fn(text, max_audio_tokens)
        warmup_time = time.perf_counter() - warmup_start
        warmup_tokens = len(pipeline._perf["steps"])
        pass_tokens.append(warmup_tokens)
        print(
            f"Warmup pass {i + 1}/{WARMUP_PASSES}: {warmup_time:.3f}s "
            f"({warmup_tokens} audio tokens, cumulative compiles={compile_count()})"
        )
        assert warmup_tokens > 0, f"warmup pass {i + 1} generated no audio tokens"

    # The pipeline owns the length; the harness only checks it held steady. A drift
    # reshapes the vocoder/latents stages and would surface as a recompile later.
    assert len(set(pass_tokens)) == 1, (
        f"the pipeline generated a different number of audio tokens across warmup "
        f"passes ({pass_tokens}) for the same text and max_audio_tokens; the "
        f"length-shaped stages will recompile. The pipeline must make the length "
        f"depend only on its config (for XTTS-v2, XTTSConfig.stop_early=False)."
    )

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
    assert len(perf["steps"]) == pass_tokens[0], (
        f"the measured pass generated {len(perf['steps'])} audio tokens against "
        f"{pass_tokens[0]} in warmup; the reported throughput is not describing "
        f"the same work the warmed-up graphs were compiled for"
    )
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
