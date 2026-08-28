# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable HiDream-I1-Full text-to-image example on Tenstorrent.

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable
demo that calls it.

The CLIP-L and CLIP-G text encoders, the Sparse-MoE MM-DiT transformer and the
VAE decoder run on the Tenstorrent backend in bf16, with the transformer
tensor-parallel sharded across the device mesh. The T5-XXL and Llama-3.1-8B
encoders and the scheduler run on CPU.

Run (multichip blackhole, qb2):
    python examples/pytorch/hidream_i1.py
"""

import torch_xla.runtime as xr

from third_party.tt_forge_models.hidream_i1.pytorch.pipeline import (
    GUIDANCE_SCALE,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    HiDreamI1Config,
    HiDreamI1Pipeline,
    save_image,
)

OUTPUT_PATH = "hidream_i1_output.png"


def main():
    xr.set_device_type("TT")

    pipeline = HiDreamI1Pipeline(config=HiDreamI1Config())
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
