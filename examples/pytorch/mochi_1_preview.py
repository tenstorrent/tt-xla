# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable Mochi-1 preview text-to-video example on Tenstorrent.

Pipeline implementation lives in ``tt_forge_models``; this is a thin demo. The
DiT (the heavy net, ~10B) runs tensor-parallel sharded on the Tenstorrent
backend; the T5-XXL text encoder, the scheduler and the VAE run on CPU.
"""

from third_party.tt_forge_models.mochi.pytorch.src.pipeline import (
    Mochi1Config,
    Mochi1Pipeline,
    save_video,
)

PROMPT = (
    "Close-up of a chameleon's eye, with its scaly skin changing color. "
    "Ultra high resolution 4k."
)
SEED = 0
NUM_INFERENCE_STEPS = 10
NUM_FRAMES = 24
FPS = 15
OUTPUT_PATH = "mochi_1_preview_output.mp4"


def main():
    pipeline = Mochi1Pipeline(
        config=Mochi1Config(
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_frames=NUM_FRAMES,
        )
    )
    pipeline.setup()

    frames = pipeline.generate(prompt=PROMPT, seed=SEED)

    save_video(frames, OUTPUT_PATH, fps=FPS)
    print(f"Saved output video to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
