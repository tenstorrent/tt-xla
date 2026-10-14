# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-speech benchmarks.

One ``test_<model>`` per model, driving a per-model pipeline through the shared
harness in ``benchmarks/tts_benchmark.py``. Mirrors the ``test_imagegen.py`` /
``imagegen_benchmark.py`` split: config here, measurement logic in ``benchmarks/``.
The headline number is the real-time factor, generated audio seconds per wall-clock
second.

XTTS-v2 needs ``coqui-tts`` + ``torchaudio``, absent from the base test
environment, so every run is wrapped in ``RequirementsManager``. Its weights are
CPML-gated behind ``COQUI_TOS_AGREED``. VibeVoice-1.5B needs neither: its
inference code is a pinned submodule and it runs on the base transformers.
"""

import inspect
import json
import os

from utils import aggregate_ttnn_perf_metrics, resolve_display_name

import third_party.tt_forge_models.xtts_v2.pytorch.loader as xtts_loader

# float32 because the submodules do not cast uniformly to bf16; opt_level 0 and
# trace disabled match the nightly e2e pipeline test.
DEFAULT_DATA_FORMAT = "float32"
DEFAULT_OPTIMIZATION_LEVEL = 0
DEFAULT_TRACE_ENABLED = False


def _write_results(results, output_file, model_info_name, perf_metrics_file):
    """Attach project metadata and serialize."""
    if not output_file:
        return
    results["project"] = "tt-forge/tt-xla"
    results["model_rawname"] = model_info_name
    aggregate_ttnn_perf_metrics(perf_metrics_file, results)
    with open(output_file, "w") as file:
        json.dump(results, file, indent=2)


def _xtts_loader_path():
    return inspect.getsourcefile(xtts_loader)


def test_xtts_v2(output_file, request):
    """XTTS-v2 end-to-end text-to-speech benchmark (real-time factor)."""
    from tests.runner.requirements import RequirementsManager

    model_info_name = "xtts-v2"
    display_name = resolve_display_name(request=request, fallback=model_info_name)
    perf_metrics_file = f"tt_xla_{display_name}_perf_metrics"

    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    with RequirementsManager.for_loader(_xtts_loader_path(), framework="torch"):
        # Deferred: importing the pipeline pulls in TTS, only present in this context.
        from benchmarks.tts_benchmark import benchmark_tts_torch_xla
        from benchmarks.xtts_v2_pipeline import (
            DEFAULT_SEED,
            MAX_AUDIO_TOKENS,
            make_bench_xtts_pipeline_cls,
        )

        from third_party.tt_forge_models.xtts_v2.pytorch.pipeline import (
            DEFAULT_TEXT,
            OUTPUT_SAMPLE_RATE,
            XTTSConfig,
            save_wav,
        )

        def build_pipeline_fn(compile_options):
            pipeline_cls = make_bench_xtts_pipeline_cls()
            config = XTTSConfig(
                text=DEFAULT_TEXT,
                max_audio_tokens=MAX_AUDIO_TOKENS,
                seed=DEFAULT_SEED,
                # Spend the full budget rather than stopping on the model's stop
                # token, so every pass sees identical shapes and cannot recompile.
                stop_early=False,
            )
            pipeline = pipeline_cls(config)
            pipeline.setup()

            def generate_fn(text, max_audio_tokens):
                config.text = text
                config.max_audio_tokens = max_audio_tokens
                return pipeline.run()

            return pipeline, generate_fn

        results = benchmark_tts_torch_xla(
            build_pipeline_fn=build_pipeline_fn,
            model_info_name=model_info_name,
            display_name=display_name,
            text=DEFAULT_TEXT,
            max_audio_tokens=MAX_AUDIO_TOKENS,
            sample_rate=OUTPUT_SAMPLE_RATE,
            optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
            trace_enabled=DEFAULT_TRACE_ENABLED,
            ttnn_perf_metrics_output_file=perf_metrics_file,
            output_wav_path="test_xtts_v2_output.wav",
            save_wav_fn=save_wav,
            data_format=DEFAULT_DATA_FORMAT,
        )

    _write_results(results, output_file, model_info_name, perf_metrics_file)


def test_vibevoice_1_5b(output_file, request):
    """VibeVoice-1.5B end-to-end text-to-speech benchmark (real-time factor).

    Unlike XTTS-v2 this needs no ``RequirementsManager``: the loader runs on the
    base test environment's transformers, and the vendored inference code is a
    pinned submodule rather than a pip dependency.

    The diffusion head and both speech connectors run on TT; the acoustic and
    semantic tokenizers and the Qwen2 LM backbone run on CPU. That hybrid is the
    target configuration for this model, so the ``lm_cpu`` share of the reported
    stage times is expected to be large — it is host work that is *meant* to be
    there, not an unaccounted gap.
    """
    from benchmarks.tts_benchmark import benchmark_tts_torch_xla
    from benchmarks.vibevoice_pipeline import (
        MAX_NEW_TOKENS,
        make_bench_vibevoice_pipeline,
    )

    from third_party.tt_forge_models.vibevoice.pytorch.pipeline import (
        DEFAULT_TEXT as VIBEVOICE_TEXT,
    )
    from third_party.tt_forge_models.vibevoice.pytorch.pipeline import (
        OUTPUT_SAMPLE_RATE as VIBEVOICE_SAMPLE_RATE,
    )
    from third_party.tt_forge_models.vibevoice.pytorch.pipeline import save_wav

    model_info_name = "vibevoice-1.5b"
    display_name = resolve_display_name(request=request, fallback=model_info_name)
    perf_metrics_file = f"tt_xla_{display_name}_perf_metrics"

    # Held so save_wav can reuse the pipeline's processor rather than loading a
    # second one off disk just to write the output file.
    built = {}

    def build_pipeline_fn(compile_options):
        pipeline, config = make_bench_vibevoice_pipeline()
        pipeline.setup()
        built["pipeline"] = pipeline

        def generate_fn(text, max_audio_tokens):
            config.text = text
            config.max_new_tokens = max_audio_tokens
            return pipeline.run()

        return pipeline, generate_fn

    results = benchmark_tts_torch_xla(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name=model_info_name,
        display_name=display_name,
        text=VIBEVOICE_TEXT,
        max_audio_tokens=MAX_NEW_TOKENS,
        sample_rate=VIBEVOICE_SAMPLE_RATE,
        optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
        trace_enabled=DEFAULT_TRACE_ENABLED,
        ttnn_perf_metrics_output_file=perf_metrics_file,
        output_wav_path="test_vibevoice_1_5b_output.wav",
        save_wav_fn=lambda wav, path: save_wav(
            wav, path, processor=built["pipeline"].processor
        ),
        data_format=DEFAULT_DATA_FORMAT,
    )

    _write_results(results, output_file, model_info_name, perf_metrics_file)
