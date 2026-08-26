# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable FLUX.1-dev text-to-image example on Tenstorrent.

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable
demo that calls it.

Every compute module runs on the Tenstorrent backend — the FluxTransformer2DModel
is tensor-parallel sharded across the device mesh, the CLIP and T5 text encoders
and the VAE decoder are replicated; only the tokenizers and the scheduler stay on
CPU.

Run (multichip blackhole, qb2):
    python examples/pytorch/flux1.py
"""

import torch_xla.runtime as xr

from third_party.tt_forge_models.flux.pytorch.pipeline import (
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    FluxConfig,
    FluxTTPipeline,
    save_image,
)

OUTPUT_PATH = "flux1_output.png"


def main():
    xr.set_device_type("TT")

    pipeline = FluxTTPipeline(config=FluxConfig())
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
