# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable VibeVoice-1.5B (microsoft/VibeVoice-1.5B) text-to-speech example.

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable
demo that calls it.

VibeVoice clones the timbre of a reference clip rather than selecting a fixed
speaker ID, so the inputs are a script and a voice-prompt wav; the demo defaults
to the clip shipped with the vendored submodule. The diffusion head and both
speech connectors run on the Tenstorrent backend; the acoustic/semantic
tokenizers and the Qwen2 LM backbone run on CPU. That hybrid is the target
configuration — see ``pipeline.LM_RESIDENCY_NOTE`` for what moving the LM needs.

Each component on device is PCC-gated per forward against a CPU twin of itself,
so the run reports whether the device path actually matched, not just that it
produced audio. Pass ``gate=False`` for a plain synthesis without the twins.

Run (single-chip wormhole, n150/n300):
    python examples/pytorch/vibevoice_1_5b.py
"""

import torch
import torch_xla.runtime as xr

from third_party.tt_forge_models.vibevoice.pytorch.pipeline import (
    DEFAULT_TEXT,
    OUTPUT_SAMPLE_RATE,
    run_vibevoice_pipeline,
)

OUTPUT_PATH = "vibevoice_1_5b_output.wav"


def main():
    xr.set_device_type("TT")

    # max_new_tokens is left at the pipeline default so generation runs until the
    # model emits EOS, i.e. the full utterance rather than a fixed budget.
    wav, pipeline = run_vibevoice_pipeline(
        output_path=OUTPUT_PATH,
        text=DEFAULT_TEXT,
        # float32: the CPU twins are the PCC reference for the device forwards,
        # so they should not themselves be quantised.
        dtype=torch.float32,
        seed=0,
    )

    duration = wav.shape[-1] / OUTPUT_SAMPLE_RATE
    print(
        f"\nSaved output audio to {OUTPUT_PATH} "
        f"({duration:.2f}s @ {OUTPUT_SAMPLE_RATE} Hz, {pipeline.steps} LM steps, "
        f"stop={'EOS' if pipeline.saw_eos else 'step cap'})"
    )
    print(f"\n{pipeline.summary()}")
    print(
        "\nPer-forward PCC is the device-correctness check. It is NOT an "
        "end-to-end acceptance criterion: waveform PCC against a CPU golden "
        "reads 0.734 on a fully correct run, because each frame's latent feeds "
        "back into the next LM step and a 3e-05 per-forward difference compounds. "
        "Listen to the wav, or transcribe it."
    )


if __name__ == "__main__":
    main()
