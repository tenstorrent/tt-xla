# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Audio-pipeline benchmarks.

Config-driven entry points (one ``test_<model>`` per model) that drive per-model
pipelines through the shared harness in ``benchmarks/audio_benchmark.py``. Same
split as ``test_imagegen.py`` / ``imagegen_benchmark.py``: model-specific config
lives here, the reusable measurement logic lives in ``benchmarks/``.

Covers models whose input or output is audio and whose generation is an
autoregressive decode loop; throughput is reported as decode tokens/second.
"""

import json

from benchmarks.audio_benchmark import benchmark_audio_torch_xla
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

from third_party.tt_forge_models.voxtral.pytorch.pipeline import (
    VoxtralConfig,
    VoxtralPipeline,
)

# Defaults shared by all audio models.
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_TRACE_ENABLED = False


def test_audio(
    build_pipeline_fn,
    model_info_name,
    output_file,
    num_generation_steps,
    model_type,
    dataset_name,
    request=None,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    save_output_fn=None,
):
    """Run an audio-pipeline benchmark with the given configuration.

    Args:
        build_pipeline_fn: Callable returning ``(pipeline, generate_fn)``;
            see ``benchmark_audio_torch_xla``.
        model_info_name: Model name for identification and reporting.
        output_file: Path to save benchmark results as JSON.
        num_generation_steps: Token budget per generation (decode-loop cap).
        model_type: Dashboard model type (e.g. "audio-to-text").
        dataset_name: Dashboard dataset name (e.g. "Audio Prompt").
        optimization_level: Optimization level (0, 1, or 2).
        trace_enabled: Enable trace.
        save_output_fn: If set, persists the steady-state output.
    """
    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    print(f"Running audio benchmark for model: {model_info_name}")
    print(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    num_generation_steps={num_generation_steps}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_audio_torch_xla(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name=model_info_name,
        display_name=resolved_display_name,
        num_generation_steps=num_generation_steps,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        model_type=model_type,
        dataset_name=dataset_name,
        save_output_fn=save_output_fn,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_voxtral_mini_3b(output_file, request):
    # Audio tower + audio/text merge run on the host (the HF merge is a
    # data-dependent masked_scatter); the 30-layer Ministral-3B LM decodes on TT.
    # 32 tokens is enough for a full answer to the sample question and bounds the
    # run: the loop has no KV cache, so every step is a full-window forward.
    num_generation_steps = 32
    answer_path = "test_voxtral_mini_3b_answer.txt"

    def build_pipeline_fn(compile_options):
        pipeline = VoxtralPipeline(
            config=VoxtralConfig(max_new_tokens=num_generation_steps)
        )
        pipeline.setup()

        def generate_fn(max_new_tokens):
            return pipeline.generate(max_new_tokens=max_new_tokens)

        return pipeline, generate_fn

    def save_output_fn(answer):
        print(f"Voxtral answer: {answer}")
        with open(answer_path, "w") as file:
            file.write(answer)
        print(f"Saved answer to {answer_path}")

    test_audio(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="Voxtral-Mini-3B-2507",
        output_file=output_file,
        request=request,
        num_generation_steps=num_generation_steps,
        model_type="audio-to-text",
        dataset_name="Audio Prompt",
        save_output_fn=save_output_fn,
    )
