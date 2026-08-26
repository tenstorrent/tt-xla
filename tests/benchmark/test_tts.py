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
CPML-gated behind ``COQUI_TOS_AGREED``.
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
