# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic text-to-speech benchmark harness for torch-xla / TT.

Sibling of ``imagegen_benchmark.py`` and ``ar_imagegen_benchmark.py``, for models
whose output is a waveform. Per-model wiring supplies a ``build_pipeline_fn`` and
the pipeline reports its own stage names, so a new TTS model needs a
``*_pipeline.py`` plus a test entry, not a new harness.

A TTS model is several compiled graphs chained by CPU orchestration rather than one
traceable forward, so the harness times the chain and reports each stage as its own
``<stage>_s`` measurement. Headline throughput is the real-time factor: generated
audio seconds per wall-clock second.

``WARMUP_PASSES`` *full* generations run before the timed one: the vocoder and
latents stages are shaped by the generated token count, so anything shorter would
compile inside the measured pass. Recompiles there are a hard error, not a warning
-- they inflate the result by an amount that reads as an ordinary regression.

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

# Full generations before the measured pass. Two is structural: pass 1's decode
# graphs take freshly-uploaded host buffers, pass 2's take pass 1's own outputs --
# different HLO at the same shapes, so it compiles again. Do not lower this while
# the pipeline reuses a buffer across runs.
WARMUP_PASSES = 2

# Minimum share of wall time the stage timers must account for. Below this they are
# not bracketing device work, almost always a missing sync_device().
MIN_STAGE_TIME_FRACTION = 0.3

# Graph reuse is judged over the second half of the decode curve; two points is the
# fewest that has a growth delta at all.
MIN_SETTLED_STEPS = 2

# Float slack when comparing summed stage times against a single wall-clock read.
STAGE_TIME_TOLERANCE_S = 1e-6

MODEL_TYPE = "Audio Generation, Text-to-Speech"
DATASET_NAME = "Text Prompt"


def compile_count() -> int:
    """Cumulative number of graph compilations observed so far."""
    data = met.metric_data("CompileTime")
    return data[0] if data else 0


def sync_device() -> None:
    """Launch any pending XLA graph and wait for it.

    torch-xla is lazy, so a timer closed when a forward returns measures tracing
    rather than device work. ``wait_device_ops()`` alone is a no-op when nothing
    has been launched, so both calls are needed.
    """
    torch_xla.sync()
    xm.wait_device_ops()


def assert_no_recompiles(before: int, after: int, what: str) -> None:
    """Fail if any graph compiled between the two counter reads."""
    print(f"| Compilations during {what}: {after - before} ({before} -> {after})")
    assert after == before, (
        f"{after - before} graph(s) compiled during {what} ({before} -> {after}); "
        f"the reported timings include compilation"
    )


def assert_decode_graph_reused(compile_curve) -> None:
    """Fail if the decode loop kept compiling instead of reusing one graph."""
    assert compile_curve, (
        "the pipeline recorded no _perf['compile_curve']; the per-model wiring "
        "must append compile_count() after each decode call"
    )
    settled = compile_curve[len(compile_curve) // 2 :]
    assert len(settled) >= MIN_SETTLED_STEPS, (
        f"too few decode steps to judge graph reuse: {compile_curve}; the settled "
        f"half needs {MIN_SETTLED_STEPS} steps to measure growth across"
    )
    tail_growth = settled[-1] - settled[0]
    assert tail_growth == 0, (
        f"the decode loop kept compiling instead of reusing its graph: "
        f"{tail_growth} new compilation(s) over the last {len(settled)} of "
        f"{len(compile_curve)} steps; cumulative compiles per step={compile_curve}"
    )


def assert_stage_timings_plausible(stages_total: float, total: float) -> None:
    """Sanity-check that the stage timers actually bracketed device work."""
    assert stages_total <= total + STAGE_TIME_TOLERANCE_S, (
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
        build_pipeline_fn: ``(compile_options) -> (pipeline, generate_fn)``, where
            ``generate_fn(text, max_audio_tokens) -> waveform`` runs one full
            synthesis and populates ``pipeline._perf``.
        model_info_name: Model name for identification and reporting.
        text: Text to synthesize.
        max_audio_tokens: Audio tokens per run. The generated length must depend
            only on this value, not on sampling, so the length-shaped stages see
            identical shapes in every pass and the RTF stays comparable.
        sample_rate: Output waveform sample rate in Hz.
        optimization_level: tt-mlir optimization level for compilation.
        trace_enabled: Whether to enable tracing.
        ttnn_perf_metrics_output_file: Base path for TTNN perf metrics files.
        display_name: Display name used for export naming / dashboard.
        output_wav_path: If set, the steady-state waveform is saved here.
        save_wav_fn: ``save_wav_fn(waveform, path)``, injected so this module does
            not depend on any model's audio stack.
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

    # Kernels compile lazily on the first forward, i.e. during warmup below.
    pipeline, generate_fn = build_pipeline_fn(options)

    # Warmup: full syntheses at the shapes the measured pass reuses.
    pass_tokens = []
    for i in range(WARMUP_PASSES):
        print(f"Starting warmup pass {i + 1}/{WARMUP_PASSES} (includes compile)...")
        warmup_start = time.perf_counter()
        last_warmup_wav = generate_fn(text, max_audio_tokens)
        warmup_time = time.perf_counter() - warmup_start
        warmup_tokens = len(pipeline._perf["steps"])
        pass_tokens.append(warmup_tokens)
        last_warmup_curve = list(pipeline._perf.get("compile_curve") or [])
        print(
            f"Warmup pass {i + 1}/{WARMUP_PASSES}: {warmup_time:.3f}s "
            f"({warmup_tokens} audio tokens, cumulative compiles={compile_count()})"
        )
        assert warmup_tokens > 0, f"warmup pass {i + 1} generated no audio tokens"

    # A drifting length reshapes the vocoder/latents stages and recompiles later.
    assert len(set(pass_tokens)) == 1, (
        f"the pipeline generated a different number of audio tokens across warmup "
        f"passes ({pass_tokens}); the generated length must depend only on "
        f"max_audio_tokens, not on what sampling yielded"
    )

    # Only place this check can bite: assert_no_recompiles below already forces the
    # measured pass's curve flat.
    assert_decode_graph_reused(last_warmup_curve)

    # Measured pass: every forward must be a cache hit. Its audio is what gets saved.
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
        f"{pass_tokens[0]} in warmup; it is not the work the graphs were compiled for"
    )
    # Wiring check only; the real one ran on warmup above.
    assert_decode_graph_reused(perf.get("compile_curve"))

    # Sampling is re-seeded per run, so the measured pass must reproduce the warmup
    # output exactly. A mismatch means stale K/V leaked across runs.
    assert tuple(wav.shape) == tuple(last_warmup_wav.shape), (
        f"measured waveform {tuple(wav.shape)} differs in shape from warmup "
        f"{tuple(last_warmup_wav.shape)}; the reused KV cache is leaking state"
    )
    max_dev = (wav.float() - last_warmup_wav.float()).abs().max().item()
    print(f"| Max |measured - warmup| sample deviation: {max_dev:.3e}")
    assert max_dev == 0.0, (
        f"the measured pass produced different audio from warmup (max sample "
        f"deviation {max_dev:.3e}); the reused KV cache is leaking state"
    )

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
    # "A sample" is one second of audio, so the dashboard's total_samples/total_time
    # is exactly the real-time factor.
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
    # One measurement per pipeline stage, named by the pipeline.
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
