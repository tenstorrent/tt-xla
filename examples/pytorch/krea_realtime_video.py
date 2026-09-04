# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable Krea Realtime Video (14B) text-to-video example on Tenstorrent.

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable
demo that calls it.

The UMT5 text encoder (run once, then freed), the CausalWan DiT (the heavy net)
and the VAE decoder run on the Tenstorrent backend — the DiT tensor-parallel
sharded across the device mesh, the VAE decoder replicated.

Run (multichip blackhole, qb2):
    python examples/pytorch/krea_realtime_video.py
"""

import torch_xla.runtime as xr
from diffusers.utils import export_to_video

from third_party.tt_forge_models.krea_realtime_video.pytorch.pipeline import (
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    KreaRealtimePipeline,
)

NUM_BLOCKS = 1  # >1 pending investigation (S64/S32 dtype mismatch in flex_attention's create_block_mask): https://github.com/tenstorrent/tt-xla/issues/5837
FPS = 24
OUTPUT_PATH = "krea_realtime_video_output.mp4"


def main():
    xr.set_device_type("TT")

    pipeline = KreaRealtimePipeline()
    pipeline.setup()

    frames = pipeline.generate(
        prompt=PROMPT,
        num_blocks=NUM_BLOCKS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    export_to_video(frames, OUTPUT_PATH, fps=FPS)
    print(f"Saved output video to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
