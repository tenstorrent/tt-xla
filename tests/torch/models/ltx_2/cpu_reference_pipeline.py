"""
Run diffusers LTX2Pipeline on CPU as a ground-truth reference.
Saves intermediate tensors and final video for comparison with TT output.

Uses 24-layer truncation (same as TT pipeline) so we can do apples-to-apples comparison.
Also runs full 48-layer for quality reference.
"""
import os, sys, time, copy
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np


def run_reference(num_layers=None, num_inference_steps=20, prompt="A dog playing a guitar on the street with a rat"):
    from diffusers import LTX2Pipeline

    print(f"Loading LTX2Pipeline (CPU)...")
    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)

    if num_layers is not None and num_layers < len(pipe.transformer.transformer_blocks):
        orig_layers = len(pipe.transformer.transformer_blocks)
        pipe.transformer.transformer_blocks = nn.ModuleList(
            list(pipe.transformer.transformer_blocks)[:num_layers]
        )
        pipe.transformer.config.num_layers = num_layers
        print(f"  Truncated transformer: {orig_layers} -> {num_layers} layers")

    pipe = pipe.to("cpu")

    print(f"Running pipeline: '{prompt}', {num_inference_steps} steps...")
    t0 = time.time()
    generator = torch.Generator("cpu").manual_seed(42)
    output = pipe(
        prompt=prompt,
        negative_prompt="",
        height=512,
        width=320,
        num_frames=49,
        num_inference_steps=num_inference_steps,
        guidance_scale=4.0,
        generator=generator,
        output_type="np",
    )
    elapsed = time.time() - t0
    print(f"Pipeline done in {elapsed:.0f}s")

    video = output.frames[0]  # [F, H, W, C] float [0,1]
    print(f"Video shape: {video.shape}, range=[{video.min():.3f},{video.max():.3f}]")

    # Convert to uint8
    frames_uint8 = (video * 255).clip(0, 255).astype(np.uint8)
    tag = f"{num_layers}L" if num_layers else "full"
    npy_path = f"/root/tt-xla/cpu_ref_{tag}_{num_inference_steps}steps.npy"
    np.save(npy_path, frames_uint8)
    print(f"Saved: {npy_path}")

    # Save as mp4
    try:
        import av
        mp4_path = npy_path.replace(".npy", ".mp4")
        container = av.open(mp4_path, mode="w")
        stream = container.add_stream("libx264", rate=24)
        stream.width = 320; stream.height = 512; stream.pix_fmt = "yuv420p"
        for f in frames_uint8:
            frame = av.VideoFrame.from_ndarray(f, format="rgb24")
            for p in stream.encode(frame): container.mux(p)
        for p in stream.encode(): container.mux(p)
        container.close()
        print(f"Saved: {mp4_path} ({os.path.getsize(mp4_path)/1024:.0f} KB)")
    except ImportError:
        print("PyAV not available, skipping mp4")

    return frames_uint8


if __name__ == "__main__":
    # Run 24-layer (same as TT) for direct comparison
    print("=" * 60)
    print("24-LAYER REFERENCE (matching TT pipeline)")
    print("=" * 60)
    frames_24 = run_reference(num_layers=24, num_inference_steps=20)

    print(f"\nStats: mean={frames_24.mean():.1f}, std={frames_24.std():.1f}")
    f0 = frames_24[0].astype(float)
    f1 = frames_24[1].astype(float)
    print(f"Frame 0-1 diff: {np.abs(f0-f1).mean():.1f}")
