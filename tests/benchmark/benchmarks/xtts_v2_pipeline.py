# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 (coqui/XTTS-v2) — benchmark-side wiring for the TTS harness.

``XTTSPipeline`` in tt-forge-models does its own per-stage ``_perf`` timing, places
every learned module on device once in ``setup()`` and never moves them again, and
reuses one KV cache across ``run()`` calls. That is the whole steady-state story,
so the measured code path is the upstream one and this module adds exactly one
thing that cannot live upstream:

* ``compile_curve`` — cumulative ``CompileTime`` after each decode call, which
  feeds the harness's ``assert_decode_graph_reused``. Reading it needs
  ``torch_xla.debug.metrics``, and tt-forge-models stays framework-agnostic, so
  it is recorded from this side.

The generated length is the pipeline's business too: ``test_tts.py`` sets
``XTTSConfig.stop_early=False``, so a run always produces the full
``max_audio_tokens`` budget and the length-shaped stages (``gpt_latents``,
``hifigan``) see identical shapes every pass without the benchmark pinning
anything.

Stages the pipeline times, in the order ``run()`` executes them:
``speaker_encoder`` -> ``conditioning`` (summed over mel chunks) -> ``gpt_prefill``
-> the audio-token decode loop (``steps``) -> ``gpt_latents`` -> ``hifigan``.
"""

# Audio-token budget for the benchmark run. With ``stop_early=False`` the decode
# loop always spends it, so the generated duration -- and therefore the real-time
# factor -- is the same from one nightly to the next. Large enough that the decode
# loop dominates the measurement rather than the one-shot stages around it.
MAX_AUDIO_TOKENS = 128
DEFAULT_SEED = 0


def make_bench_xtts_pipeline_cls():
    """Build the instrumented ``XTTSPipeline`` subclass.

    Deferred into a function because importing the pipeline pulls in ``TTS``, which
    is only installed inside the ``RequirementsManager`` context the caller opens.
    """
    import torch
    from benchmarks.tts_benchmark import compile_count, sync_device

    from third_party.tt_forge_models.xtts_v2.pytorch.pipeline import XTTSPipeline

    class _CompileCounted(torch.nn.Module):
        """Records the cumulative compile count after each call of a stage.

        A wrapper rather than a forward hook: ``XTTSPipeline.setup()`` has already
        ``torch.compile``d each stage, and registering hooks on a compiled module
        can invalidate its guards and trigger a recompile -- the exact thing this
        benchmark asserts against. The wrapper sits outside the compiled region, so
        it observes the stage without perturbing it.

        The wrapper holds no parameters of its own, so it does not disturb the
        placement ``setup()`` already did for the module it wraps.
        """

        def __init__(self, inner, record):
            super().__init__()
            self.inner = inner
            self._record = record

        def forward(self, *args, **kwargs):
            out = self.inner(*args, **kwargs)
            # A compile count is only meaningful once the graph has actually been
            # launched. The pipeline forces that a moment later with its
            # ``.to("cpu")``, but the counter has to be read after the sync, not
            # before it. The sync sits inside the region the pipeline is timing,
            # so it does not change what that timer measures.
            sync_device()
            self._record()
            return out

    class BenchXTTSPipeline(XTTSPipeline):
        """``XTTSPipeline`` that records a per-step compile curve."""

        def _reset_perf(self):
            super()._reset_perf()
            self._perf["compile_curve"] = []

        def _record_compile_count(self):
            self._perf["compile_curve"].append(compile_count())

        def setup(self):
            super().setup()
            self.decode_step = _CompileCounted(
                self.decode_step, self._record_compile_count
            )

    return BenchXTTSPipeline
