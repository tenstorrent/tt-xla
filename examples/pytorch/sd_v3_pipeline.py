# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable Stable Diffusion 3 Medium text-to-image example on Tenstorrent.

The reusable pipeline implementation is shared with the image-gen benchmark
(``tests/benchmark/test_imagegen.py``) and the nightly test, and lives in
``tt_forge_models`` — this script is just a thin runnable demo so the
implementation isn't duplicated in ``examples/``.

The MMDiT transformer (the heavy net) runs on the Tenstorrent backend via
``torch.compile(backend="tt")``; the three text encoders (two CLIP + T5), the
scheduler and the VAE run on CPU.
"""

from third_party.tt_forge_models.stable_diffusion_3.pytorch.pipeline import (
    SD3Config,
    SD3Pipeline,
    save_image,
)

PROMPT = "An astronaut riding a green horse"
NEGATIVE_PROMPT = ""
GUIDANCE_SCALE = 7.0
NUM_INFERENCE_STEPS = 28
SEED = 42
OUTPUT_PATH = "sd_v3_output.png"


def main():
    pipeline = SD3Pipeline(config=SD3Config())
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
