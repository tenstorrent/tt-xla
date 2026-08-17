# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable XTTS-v2 (coqui/XTTS-v2) text-to-speech example on Tenstorrent.

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable
demo that calls it.

Every learned nn.Module runs on the Tenstorrent backend -- the speaker encoder,
the GPT prefill and KV-cached decode step, the latents head, and the HiFi-GAN
decoder; tokenization and the mel/audio pre- and post-processing run on CPU.

Needs ``coqui-tts`` + ``torchaudio`` and the CPML-gated XTTS-v2 weights
(``COQUI_TOS_AGREED``, which the pipeline sets on the caller's behalf):

    pip install -r third_party/tt_forge_models/xtts_v2/pytorch/requirements.txt

Run (single-chip wormhole, n150/n300):
    python examples/pytorch/xtts_v2_pipeline.py
"""

import torch_xla.runtime as xr

from third_party.tt_forge_models.xtts_v2.pytorch.pipeline import (
    DEFAULT_LANGUAGE,
    DEFAULT_TEXT,
    OUTPUT_SAMPLE_RATE,
    run_xtts_pipeline,
)

OUTPUT_PATH = "xtts_v2_output.wav"


def main():
    xr.set_device_type("TT")

    # max_audio_tokens is left at the pipeline default so the decode loop runs
    # until the model emits its stop token, i.e. the full utterance.
    wav = run_xtts_pipeline(
        output_path=OUTPUT_PATH,
        text=DEFAULT_TEXT,
        language=DEFAULT_LANGUAGE,
        seed=0,
    )

    duration = wav.shape[-1] / OUTPUT_SAMPLE_RATE
    print(
        f"Saved output audio to {OUTPUT_PATH} "
        f"({duration:.2f}s @ {OUTPUT_SAMPLE_RATE} Hz)"
    )


if __name__ == "__main__":
    main()
