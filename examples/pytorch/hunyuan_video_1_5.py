# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable HunyuanVideo 1.5 (480p t2v base) text-to-video example on Tenstorrent.

Pipeline implementation lives in ``tt_forge_models``; this is a thin demo. The
DiT (the heavy net) and the Qwen2.5-VL text encoder run tensor-parallel sharded
on the Tenstorrent backend, with the ByT5 glyph encoder replicated there; the
scheduler and the VAE run on CPU.
"""

import os
from pathlib import Path

from third_party.tt_forge_models.hunyuan_1_5.pytorch.src.pipeline import (
    HunyuanVideo15Config,
    HunyuanVideo15Pipeline,
    save_video,
)

# The double-quoted span is what routes text through text_encoder_2 (the ByT5
# glyph encoder); without it the pipeline feeds the DiT zero glyph embeds.
PROMPT = 'A girl holding a paper with words "Hello, world!"'
SEED = 42
NUM_INFERENCE_STEPS = 10
NUM_FRAMES = 25
FPS = 15

# Default to `generated/` beside this file rather than the cwd, so CI knows where
# to find the video (see the "Upload Generated Media" step in call-test.yml).
# Override with TT_EXAMPLE_OUTPUT_DIR to write somewhere else.
OUTPUT_DIR = Path(
    os.environ.get("TT_EXAMPLE_OUTPUT_DIR", Path(__file__).parent / "generated")
)
OUTPUT_PATH = OUTPUT_DIR / "hunyuan_video_1_5_output.mp4"


def main():
    pipeline = HunyuanVideo15Pipeline(
        config=HunyuanVideo15Config(
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_frames=NUM_FRAMES,
        )
    )
    pipeline.setup()

    frames = pipeline.generate(prompt=PROMPT, seed=SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(OUTPUT_PATH), fps=FPS)
    print(f"Saved output video to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
