# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 (coqui/XTTS-v2) — benchmark-side pipeline for the TTS harness.

Thin instrumentation layer over the shared ``XTTSPipeline`` in tt-forge-models: it
adds the per-stage timing the harness needs and nothing else, so the measured code
path stays the one the nightly e2e test validates. The upstream ``run()`` and
decode loop are inherited untouched.

Why this file exists at all: pipelines that populate ``_perf`` themselves need no
benchmark-side wrapper -- ``test_imagegen.py`` builds ``HunyuanImage21Pipeline``,
``SD15Pipeline`` and ``SD3Pipeline`` straight from tt-forge-models. ``XTTSPipeline``
does not, because it was written for the PCC test, so the timing lives here
instead. If it ever grows its own ``_perf`` the way HunyuanImage-2.1 did (inline
``perf_counter`` around each stage, reusing the ``cpu()`` calls that already force
the XLA sync), this module can be deleted and ``test_tts.py`` can import
``XTTSPipeline`` directly.

Stages timed, in the order ``run()`` executes them:
``speaker_encoder`` -> ``conditioning`` (summed over mel chunks) -> ``gpt_prefill``
-> the audio-token decode loop -> ``gpt_latents`` -> ``hifigan``.
"""

import time
from typing import Optional

# Model-independent defaults for the benchmark run. 128 audio tokens is ~5.46s of
# 24 kHz audio: long enough that the decode loop dominates the measurement, and
# fixed so the generated duration -- and therefore the real-time factor -- is
# comparable from one nightly to the next.
MAX_AUDIO_TOKENS = 128
DEFAULT_SEED = 0

# Stages kept resident on device between passes. These are exactly the wrappers
# over the ``xtts.gpt`` tree, which the pipeline otherwise shuttles on and off
# the device three separate times per run (once each for conditioning, the
# decode loop and gpt_latents) -- and those are measurably the only stages that
# recompile every pass. ``speaker_encoder`` and ``hifigan`` are deliberately
# absent: they live in the ``hifigan_decoder`` tree, whose weights the CPU-side
# ``_speaker_mel`` step uses, and they do not recompile anyway.
PINNED_STAGES = frozenset({"conditioning", "gpt_latents"})


def make_bench_xtts_pipeline_cls():
    """Build the instrumented ``XTTSPipeline`` subclass.

    Deferred into a function because importing the pipeline pulls in ``TTS``, which
    is only installed inside the ``RequirementsManager`` context. Same reason
    ``tests/torch/models/xtts_v2/test_xtts_v2_pipeline.py`` defers its own subclass.
    """
    import torch
    from benchmarks.tts_benchmark import compile_count, sync_device

    from third_party.tt_forge_models.xtts_v2.pytorch.pipeline import XTTSPipeline

    class _TimedStage(torch.nn.Module):
        """Times one stage's forward, including the XLA sync that follows it.

        A wrapper rather than a forward hook: ``XTTSPipeline.setup()`` has already
        ``torch.compile``d each stage, and registering hooks on a compiled module
        can invalidate its guards and trigger a recompile -- the exact thing this
        benchmark asserts against. The wrapper sits outside the compiled region, so
        it brackets the stage without perturbing it.

        ``.to(device)`` on the wrapper moves the wrapped module, so the pipeline's
        per-stage device shuttling keeps working unchanged.
        """

        def __init__(self, inner, record, pin=False):
            super().__init__()
            self.inner = inner
            self._record = record
            # Whether to keep this stage resident on device once placed there.
            # Only safe for stages whose parameters no CPU-side step touches --
            # see PINNED_STAGES.
            self._pin = pin
            self._pinned = False

        def to(self, *args, **kwargs):
            """Optionally keep the stage resident once it is on device.

            Re-uploading a stage's weights hands its compiled graph new
            parameter tensors, which recompiles it inside the measured pass.
            Pinning is opt-in per stage because the pipeline interleaves CPU and
            device work over shared parameter subtrees: pinning the
            ``hifigan_decoder`` tree, for instance, breaks ``_speaker_mel``,
            which runs ``speaker_encoder.torch_spec`` on CPU.
            """
            device = torch._C._nn._parse_to(*args, **kwargs)[0]
            if self._pinned and device is not None and device.type == "cpu":
                return self
            moved = super().to(*args, **kwargs)
            if self._pin and device is not None and device.type != "cpu":
                self._pinned = True
            return moved

        def forward(self, *args, **kwargs):
            start = time.perf_counter()
            out = self.inner(*args, **kwargs)
            sync_device()
            self._record(time.perf_counter() - start)
            return out

    class BenchXTTSPipeline(XTTSPipeline):
        """``XTTSPipeline`` that reports per-stage timings via ``_perf``."""

        # Set by the harness between the warmup and steady-state passes so the
        # token-count-shaped stages (gpt_latents, hifigan) see identical shapes in
        # both passes and cannot recompile mid-measurement.
        force_num_tokens: Optional[int] = None

        def __init__(self, config):
            super().__init__(config)
            self.force_num_tokens = None
            self._kv_caches = {}
            self._reset_perf()

        def _make_static_cache(self, max_cache_len, device):
            """Build the KV cache once and reuse it across passes.

            Upstream allocates a fresh ``StaticCache`` on every generation. That
            hands the decode graph brand-new buffers each run, so the measured
            pass recompiles the prefill and decode graphs even though nothing
            about their shapes changed. ``llm_benchmark.py`` has the same
            requirement and solves it the same way: keep one cache across the
            warmup and timed runs and reset its write index in between (see its
            "Reset cumulative_length to 0" block).

            Only the write index is reset -- the stale K/V bytes past it are
            never read, because the attention mask admits exactly the slots the
            current run has written.
            """
            key = (max_cache_len, str(device))
            cache = self._kv_caches.get(key)
            if cache is None:
                cache = super()._make_static_cache(max_cache_len, device)
                self._kv_caches[key] = cache
                return cache
            for layer in cache.layers:
                if getattr(layer, "cumulative_length", None) is not None:
                    layer.cumulative_length.zero_()
            return cache

        def _reset_perf(self):
            self._perf = {
                "components": {},
                "steps": [],
                "step_metric_name": "audio_token_step",
                "total": None,
                "audio_samples": 0,
                "text_tokens": 0,
                "compile_curve": [],
            }
            self._decode_calls = []

        def _component_recorder(self, name):
            """Accumulator for a stage; ``conditioning`` runs once per mel chunk."""

            def record(seconds):
                components = self._perf["components"]
                components[name] = components.get(name, 0.0) + seconds

            return record

        def _record_decode_call(self, seconds):
            self._decode_calls.append(seconds)
            # Cumulative compilations after each decode call. A flat tail is what
            # shows the loop settling onto one reused graph.
            self._perf["compile_curve"].append(compile_count())

        def setup(self):
            super().setup()
            for name in ("speaker_encoder", "conditioning", "gpt_latents", "hifigan"):
                setattr(
                    self,
                    name,
                    _TimedStage(
                        getattr(self, name),
                        self._component_recorder(name),
                        pin=name in PINNED_STAGES,
                    ),
                )
            self.decode_step = _TimedStage(
                self.decode_step, self._record_decode_call, pin=True
            )
            self._memoize_prefix_embeddings()

        def _memoize_prefix_embeddings(self):
            """Compute the GPT prefix embedding once and reuse it every pass.

            ``_generate_codes_tt`` calls ``gpt.compute_embeddings(...)`` on CPU
            tensors partway through the run, which is the one thing that forces
            the gpt tree back onto the host and so blocks pinning. Its inputs
            (the conditioning latent and the text tokens) are identical on every
            pass of a benchmark, so the result is too: run it once, then skip it
            and let the cached ``cached_prefix_emb`` stand.
            """
            gpt = self.xtts.gpt
            original = gpt.compute_embeddings

            def compute_embeddings_once(*args, **kwargs):
                if getattr(gpt.gpt_inference, "cached_prefix_emb", None) is not None:
                    return None
                return original(*args, **kwargs)

            gpt.compute_embeddings = compute_embeddings_once

        def _generate_codes_tt(self, gpt_cond_latent, text_tokens):
            self._perf["text_tokens"] = int(text_tokens.shape[-1])
            self._decode_calls = []

            codes = super()._generate_codes_tt(gpt_cond_latent, text_tokens)

            if self.force_num_tokens is not None:
                codes = codes[:, : self.force_num_tokens]

            # The loop makes one call per token plus the prefill; the first call is
            # the prefill graph, the rest are the reused decode graph.
            calls = self._decode_calls
            if calls:
                self._perf["components"]["gpt_prefill"] = calls[0]
                self._perf["steps"] = calls[1:]
            return codes

        def run(self):
            self._reset_perf()
            start = time.perf_counter()
            wav = super().run()
            self._perf["total"] = time.perf_counter() - start
            self._perf["audio_samples"] = int(wav.shape[-1])
            return wav

    return BenchXTTSPipeline
