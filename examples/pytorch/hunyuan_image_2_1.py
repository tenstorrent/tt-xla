# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable HunyuanImage 2.1 (Distilled) text-to-image example on Tenstorrent.

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable
demo that calls it.

Only the MMDiT transformer (the heavy net) runs on the Tenstorrent backend —
tensor-parallel sharded across the device mesh; the Qwen2.5-VL and ByT5 text
encoders, the scheduler, and the VAE run on CPU.

Run (multichip blackhole, qb2):
    python examples/pytorch/hunyuan_image_2_1.py
"""

import torch_xla.runtime as xr

from third_party.tt_forge_models.hunyuan_image_2_1.pytorch.pipeline import (
    DISTILLED_GUIDANCE_SCALE,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    HunyuanImage21Config,
    HunyuanImage21Pipeline,
    save_image,
)

OUTPUT_PATH = "hunyuan_image_2_1_output.png"


def main():
    xr.set_device_type("TT")

    pipeline = HunyuanImage21Pipeline(config=HunyuanImage21Config())
    pipeline.setup()

    image = pipeline.generate(
        prompt=PROMPT,
        distilled_guidance_scale=DISTILLED_GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    save_image(image, OUTPUT_PATH)
    print(f"Saved output image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
