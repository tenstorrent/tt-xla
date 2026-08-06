# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 (coqui/XTTS-v2) — benchmark-side wiring for the TTS harness.

``XTTSPipeline`` in tt-forge-models handles the steady state itself: per-stage
``_perf`` timing, all modules placed on device once in ``setup()``, one reused KV
cache, and a fixed generated length via ``XTTSConfig.stop_early=False``. So the
measured path is the upstream one and this module adds only ``compile_curve`` --
cumulative ``CompileTime`` after each decode call, which feeds the harness's
``assert_decode_graph_reused`` and needs ``torch_xla.debug.metrics``.

Stages the pipeline times, in the order ``run()`` executes them:
``speaker_encoder`` -> ``conditioning`` (summed over mel chunks) -> ``gpt_prefill``
-> the audio-token decode loop (``steps``) -> ``gpt_latents`` -> ``hifigan``.
"""

# Audio-token budget. With stop_early=False the loop always spends it, so the
# duration -- and the real-time factor -- is stable nightly to nightly.
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

        A wrapper rather than a forward hook: the stage is already
        ``torch.compile``d, and hooks on a compiled module can invalidate its
        guards and trigger the very recompile this benchmark asserts against. It
        holds no parameters, so it does not disturb the placement from ``setup()``.
        """

        def __init__(self, inner, record):
            super().__init__()
            self.inner = inner
            self._record = record

        def forward(self, *args, **kwargs):
            out = self.inner(*args, **kwargs)
            # The counter is only meaningful once the graph has launched, so sync
            # before reading it. This sits inside the region the pipeline times.
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
