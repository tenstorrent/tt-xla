# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable HunyuanVideo 1.5 (480p t2v distilled) text-to-video example on Tenstorrent.

Pipeline implementation lives in ``tt_forge_models``; this is a thin demo. The
DiT (the heavy net) runs tensor-parallel sharded on the Tenstorrent backend;
the text encoders (Qwen2.5-VL + ByT5), the scheduler and the VAE run on CPU.
"""

from third_party.tt_forge_models.hunyuan_1_5.pytorch.src.pipeline import (
    HunyuanVideo15Config,
    HunyuanVideo15Pipeline,
    save_video,
)

PROMPT = "a cat sitting on a boat"
SEED = 42
NUM_INFERENCE_STEPS = 10
NUM_FRAMES = 25
FPS = 15
OUTPUT_PATH = "hunyuan_video_1_5_output.mp4"


def main():
    pipeline = HunyuanVideo15Pipeline(
        config=HunyuanVideo15Config(
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
