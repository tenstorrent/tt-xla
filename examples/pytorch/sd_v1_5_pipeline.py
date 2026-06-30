# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable Stable Diffusion 1.5 text-to-image example on Tenstorrent.

The reusable pipeline implementation is shared with the image-gen benchmark
(``tests/benchmark/test_imagegen.py``) and the nightly test, and lives in
``tt_forge_models`` — this script is just a thin runnable demo so the
implementation isn't duplicated in ``examples/``.

The UNet (the heavy net) runs on the Tenstorrent backend via
``torch.compile(backend="tt")``; the precision-sensitive CLIP text encoder, the
scheduler and the VAE run on CPU.
"""

from third_party.tt_forge_models.stable_diffusion_1_5.pytorch.pipeline import (
    SD15Config,
    SD15Pipeline,
    save_image,
)

PROMPT = "a photo of a cat"
NEGATIVE_PROMPT = ""
CFG_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
SEED = 42
OUTPUT_PATH = "sd_v1_5_output.png"


def main():
    pipeline = SD15Pipeline(config=SD15Config())
    pipeline.setup()

    image = pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        cfg_scale=CFG_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    save_image(image, OUTPUT_PATH)
    print(f"Saved output image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
