# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable Qwen-Image text-to-image example on Tenstorrent.

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable
demo that calls it.

Every compute module runs on the Tenstorrent backend: the Qwen2.5-VL text encoder
and the QwenImage MMDiT transformer are both tensor-parallel sharded across the
device mesh, and the VAE decoder is replicated. Only the tokenizer and the
scheduler stay on host.

Run (multichip blackhole, qb2):
    python examples/pytorch/qwen_image.py
"""

import torch_xla.runtime as xr

from third_party.tt_forge_models.qwen_image.pytorch.pipeline import (
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    QwenImageConfig,
    QwenImagePipeline,
    save_image,
)

OUTPUT_PATH = "qwen_image_output.png"


def main():
    xr.set_device_type("TT")

    pipeline = QwenImagePipeline(config=QwenImageConfig())
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
