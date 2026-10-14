# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-1.5B nightly end-to-end text-to-speech pipeline test.

Runs the full ``generate()`` path — text + voice prompt in, waveform out — with
the diffusion head and both speech connectors resident on TT and the acoustic /
semantic tokenizers and the LM backbone on CPU. That split is the target shape
rather than a workaround; see ``pipeline.LM_RESIDENCY_NOTE`` for why the LM is
not in it.

Three gates, in increasing order of what they can catch:

1. the output artifact is a finite, non-empty 24 kHz waveform;
2. every forward of every TT-resident component matches a CPU twin of itself
   (per-forward PCC, accumulated in float64);
3. **the generated audio transcribes back to the requested sentence.**

(3) is the one that is both necessary and sufficient, and this model is the
reason to say so out loud. The acceptance-criteria question has failed here in
both directions:

* A broken run — fluent, well-formed, correctly-scaled speech that said nothing
  like the input text — passed every signal statistic there is (finite values,
  amplitude range, RMS, near-silent-frame fraction). Those are not acceptance
  criteria for this model.
* A fully **correct** run fails waveform PCC. Per-forward PCC of 0.99997 lands at
  a waveform PCC of 0.734 against a CPU golden while being the same length,
  stopping at the same step and transcribing identically: there are
  ``ddpm_steps`` solver steps per acoustic frame and each frame's latent feeds
  back through the connectors into the next LM step, so a 3e-05 per-forward
  difference compounds across the autoregressive frames.

So there is deliberately **no waveform-PCC assert here**, and adding one would
fail a correct run. For any model with a stochastic sampling loop feeding an
autoregressive one, the semantic check is the only gate stable under correct
numerics.
"""

import pytest
import torch
from infra import RunMode
from utils import BringupStatus, Category, ModelGroup

# Per-forward correlation floor for each TT-resident component against its CPU
# twin. The measured minimum across a passing run is 0.999939.
PCC_THRESHOLD = 0.99

# Word-error-rate ceiling for the transcript gate.
#
# The absolute number looks loose and is not. The reference sentence normalises
# to 11 words, three of which no ASR renders as one token — "Tenstorrent" comes
# back as "Ten Store", "VibeVoice" as "Vibe Voice", "end to end" as
# "end-to-end" — so a *fully correct* waveform reads 0.45-0.64, and the
# transformers-4.51.3 reference implementation scores 0.64 on its own output.
# The failure this gate exists to catch reads 20-51. The margin is ~20x, and
# tightening the threshold would fail correct runs rather than catch more.
MAX_WER = 0.9

ASR_MODEL = "openai/whisper-small.en"
ASR_SAMPLE_RATE = 16000


def _normalise(text):
    """Lowercase, strip punctuation, split on whitespace."""
    import re

    return re.sub(r"[^a-z0-9 ]", "", text.lower()).split()


def _wer(reference, hypothesis):
    """Standard Levenshtein word error rate."""
    import numpy as np

    ref, hyp = _normalise(reference), _normalise(hypothesis)
    distance = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    distance[:, 0] = np.arange(len(ref) + 1)
    distance[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            distance[i, j] = min(
                distance[i - 1, j] + 1,
                distance[i, j - 1] + 1,
                distance[i - 1, j - 1] + cost,
            )
    return distance[len(ref), len(hyp)] / max(len(ref), 1)


def _transcribe(wav_path):
    """Transcribe a wav with Whisper on CPU.

    Resampling is done with scipy rather than the ASR pipeline's own resampler,
    which needs torchaudio (absent from the base test environment).
    """
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly
    from transformers import pipeline as hf_pipeline

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL,
        dtype=torch.float32,
        device="cpu",
        chunk_length_s=30,
    )

    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != ASR_SAMPLE_RATE:
        divisor = gcd(sample_rate, ASR_SAMPLE_RATE)
        audio = resample_poly(
            audio, ASR_SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype("float32")
    return asr({"raw": audio, "sampling_rate": ASR_SAMPLE_RATE})["text"].strip()


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="VibeVoice_1_5B_Pipeline",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_vibevoice_e2e_pipeline(tmp_path):
    """Full VibeVoice chain on TT: waveform validity, per-forward PCC, transcript."""
    pytest.importorskip("torch_xla")
    import torch_xla.runtime as xr

    from third_party.tt_forge_models.vibevoice.pytorch.pipeline import (
        DEFAULT_REFERENCE_TRANSCRIPT,
        OUTPUT_SAMPLE_RATE,
        VibeVoiceConfig,
        VibeVoicePipeline,
        save_wav,
    )

    xr.set_device_type("TT")
    if xr.global_runtime_device_count() < 1:
        pytest.skip("No TT device available")

    pipeline = VibeVoicePipeline(
        VibeVoiceConfig(
            # float32 throughout: the CPU twin is the PCC golden for the TT run,
            # so it must not itself be quantised, and the acoustic decoder reads
            # 0.988 in bf16 against 0.998 in fp32 (precision in a deep causal-conv
            # stack, not a compiler defect).
            dtype=torch.float32,
            gate=True,
            pcc_threshold=PCC_THRESHOLD,
        )
    ).setup()
    wav = pipeline.run()

    output_path = str(tmp_path / "vibevoice_e2e_output.wav")
    save_wav(wav, output_path, processor=pipeline.processor)

    # 1) Output artifact validity. Necessary, nowhere near sufficient — the
    #    broken run this model produced in August satisfied all of this.
    assert wav.shape[-1] > 0, "pipeline produced an empty waveform"
    assert torch.isfinite(wav).all(), "waveform contains non-finite samples"
    seconds = wav.shape[-1] / OUTPUT_SAMPLE_RATE
    # print, not logger: the loguru sinks are torn down around tests (see
    # conftest's newline_logger), and these numbers are what the run is for.
    print(
        f"[e2e] {wav.shape[-1]} samples = {seconds:.2f}s over {pipeline.steps} "
        f"LM steps, stop={'EOS' if pipeline.saw_eos else 'STEP CAP (truncated)'}",
        flush=True,
    )

    # 2) Per-forward PCC for every component resident on TT. Raised inside the
    #    residency wrapper the moment a forward drops below the threshold, so
    #    reaching here means every forward passed; assert the summary anyway so
    #    a residency that recorded nothing cannot pass silently.
    print(f"[e2e] {pipeline.summary()}", flush=True)
    for residency in pipeline.residencies:
        assert residency.pccs, (
            f"{residency.name} is resident on TT but recorded no gated forwards; "
            f"the residency wrapper was bypassed"
        )
    worst = pipeline.worst_pcc()
    assert (
        worst >= PCC_THRESHOLD
    ), f"worst per-forward PCC {worst:.6f} < {PCC_THRESHOLD}"

    # 3) The transcript gate. Deliberately last and deliberately the only content
    #    check — see the module docstring for why waveform PCC is not used.
    try:
        transcript = _transcribe(output_path)
    except Exception as exc:  # ASR weights unreachable -> the gate did not run
        pytest.skip(
            f"could not run the transcript gate ({type(exc).__name__}: {exc}); "
            f"the waveform and PCC checks passed but CONTENT WAS NOT VERIFIED, "
            f"which is the only check that catches this model's failure mode"
        )

    score = _wer(DEFAULT_REFERENCE_TRANSCRIPT, transcript)
    print(
        f"[e2e] reference:  {DEFAULT_REFERENCE_TRANSCRIPT!r}\n"
        f"[e2e] transcript: {transcript!r}\n"
        f"[e2e] WER={score:.2f} (gate <= {MAX_WER})",
        flush=True,
    )
    assert score <= MAX_WER, (
        f"generated audio does not say the requested sentence: WER {score:.2f} > "
        f"{MAX_WER}\n  reference:  {DEFAULT_REFERENCE_TRANSCRIPT!r}\n"
        f"  transcript: {transcript!r}"
    )
