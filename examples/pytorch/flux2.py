# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable FLUX.2-dev text-to-image example on Tenstorrent.

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable
demo that calls it.

Every compute module runs on the Tenstorrent backend — the Mistral3 text encoder
(~24B) and the Flux2 transformer (~32B) tensor-parallel sharded across the device
mesh, the VAE decoder replicated; only the tokenizer and the scheduler stay on
CPU.

Run (multichip blackhole, qb2):
    python examples/pytorch/flux2.py
"""

import torch_xla.runtime as xr

from third_party.tt_forge_models.flux2.pytorch.pipeline import (
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    Flux2Config,
    Flux2TTPipeline,
    save_image,
)

OUTPUT_PATH = "flux2_output.png"


def main():
    xr.set_device_type("TT")

    pipeline = Flux2TTPipeline(config=Flux2Config())
    pipeline.setup()

    image = pipeline.generate(
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    save_image(image, OUTPUT_PATH)
    print(f"Saved output image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
