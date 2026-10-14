# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-1.5B (microsoft/VibeVoice-1.5B) — benchmark-side wiring for the TTS harness.

``VibeVoicePipeline`` in tt-forge-models already populates the whole ``_perf``
contract when built with ``collect_perf=True``, ``compile_curve`` included, so
this module is configuration rather than instrumentation.

Stage decomposition, in ``run()`` order: ``tokenizer_encode`` (voice-prompt
conditioning) -> the per-acoustic-frame loop (``steps``: ``ddpm_steps`` diffusion
head forwards on TT, both connectors on TT, and the CPU LM decode for that step)
-> ``audio_decode`` (latents to waveform).

Two settings differ from the nightly test and both are deliberate:

* ``gate=False``. The nightly test runs a CPU twin of every resident component in
  lockstep to PCC-gate each forward. That roughly doubles the wall time and would
  be measured as if it were device work — the 85.9 s / RTF 22.20x first reading
  of this model was exactly that mistake.
* ``max_new_tokens`` is pinned rather than left to EOS. The harness requires the
  generated length to be identical across passes; pinning it makes that a
  property of the configuration instead of something the run has to reproduce.
"""

# Acoustic frames per run. Each is one LM decode step plus ddpm_steps diffusion
# head forwards, and yields ~3200 output samples at 24 kHz, so 32 frames is
# roughly 4 s of audio — the same order as the natural EOS length of the default
# script, which keeps the number comparable to the nightly test's.
MAX_NEW_TOKENS = 32

# Re-applied before every generate(). The diffusion loop draws noise per frame
# and that noise feeds back into the next LM step, so without a fixed seed the
# generated length itself drifts run to run and the harness's equal-length
# assertion fires.
DEFAULT_SEED = 0


def make_bench_vibevoice_pipeline():
    """Build a benchmark-configured ``VibeVoicePipeline`` (not yet set up)."""
    import torch

    from third_party.tt_forge_models.vibevoice.pytorch.pipeline import (
        DEFAULT_COMPONENTS,
        VibeVoiceConfig,
        VibeVoicePipeline,
    )

    config = VibeVoiceConfig(
        components=DEFAULT_COMPONENTS,
        # float32: the acoustic decoder reads 0.988 in bf16 against 0.998 in
        # fp32, so the pipeline is benchmarked in the precision it is correct in.
        dtype=torch.float32,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=DEFAULT_SEED,
        gate=False,
        collect_perf=True,
    )
    return VibeVoicePipeline(config), config
