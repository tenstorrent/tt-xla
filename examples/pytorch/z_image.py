# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable Z-Image (Tongyi-MAI/Z-Image) text-to-image example on Tenstorrent.

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable
demo that calls it.

Every compute module runs on a single Blackhole chip — the Qwen3 text encoder,
the ZImageTransformer2DModel denoise loop and the AutoencoderKL decoder are each
placed, used and freed in turn; the scheduler and tokenizer stay on CPU. Nothing
is sharded, so this uses one chip even on a multi-chip host. Its weights exceed
the DRAM a single Wormhole chip provides, so it OOMs there and is Blackhole-only.

optimization_level=1 keeps GroupNorm as native ttnn.group_norm so the VAE decode
at 1280x720 does not OOM (issue #4755).

Run (single Blackhole chip, e.g. p150):
    python examples/pytorch/z_image.py
"""

import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.z_image.pytorch.pipeline import (
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    ZImageConfig,
    ZImageTTPipeline,
    save_image,
)

OUTPUT_PATH = "z_image_output.png"


def main():
    xr.set_device_type("TT")
    # opt_level=1 keeps GroupNorm native so the 1280x720 VAE decode fits.
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    pipeline = ZImageTTPipeline(config=ZImageConfig())
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
