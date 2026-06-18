# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable BRIA 2.3 text-to-image example on Tenstorrent.

The reusable pipeline implementation is shared with the image-gen benchmark and
lives in ``tt_forge_models`` — this script is just a thin runnable demo so the
implementation isn't duplicated in ``examples/``.

BRIA 2.3 is an SDXL-class model: the UNet (the heavy net) runs on the
Tenstorrent backend via ``torch.compile(backend="tt")``; the two CLIP text
encoders, the scheduler and the VAE run on CPU.

Note: ``briaai/BRIA-2.3`` is a gated Hugging Face repo — accept its terms and
authenticate (``huggingface-cli login``) before running.
"""

from third_party.tt_forge_models.bria_2_3.pytorch.pipeline import (
    Bria23Config,
    Bria23Pipeline,
    save_image,
)

PROMPT = (
    "A portrait of a Beautiful and playful ethereal singer, "
    "golden designs, highly detailed, blurry background"
)
NEGATIVE_PROMPT = ""
GUIDANCE_SCALE = 5.0
NUM_INFERENCE_STEPS = 50
SEED = 42
OUTPUT_PATH = "bria_2_3_output.png"


def main():
    pipeline = Bria23Pipeline(config=Bria23Config())
    pipeline.setup()

    image = pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    save_image(image, OUTPUT_PATH)
    print(f"Saved output image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
