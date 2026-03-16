# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 T2V (Text-to-Video) generation on a single Tenstorrent device.

Each pipeline stage runs as a **separate process** to get a clean device state
(tt-metal segfaults when running 3+ compiled programs sequentially on the same
device). Intermediate results are saved/loaded via .pt files in wan_artifacts/.

Usage — run each stage separately:
    python wan22_1x4_mesh.py --stage encode --small
    python wan22_1x4_mesh.py --stage denoise --small --num_inference_steps 2
    python wan22_1x4_mesh.py --stage decode --small

Or run all stages sequentially (each in its own process):
    python wan22_1x4_mesh.py --stage all --small --num_inference_steps 2

Configuration matches the validated per-component tests in
tests/torch/models/wan/.

Requirements:
    pip install diffusers transformers accelerate imageio imageio-ffmpeg
"""

import argparse
import os
import time

import torch
import torch.nn as nn

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

DEFAULT_NUM_FRAMES = 5
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_FPS = 16

SMALL_HEIGHT = 64
SMALL_WIDTH = 64

VAE_SCALE_TEMPORAL = 4
VAE_SCALE_SPATIAL = 8
LATENT_CHANNELS = 16

ARTIFACTS_DIR = "wan_artifacts"
ENC_PATH = os.path.join(ARTIFACTS_DIR, "encoder_hidden_states.pt")
LATENTS_PATH = os.path.join(ARTIFACTS_DIR, "latents.pt")
VIDEO_PATH = os.path.join(ARTIFACTS_DIR, "video.pt")


# ---------------------------------------------------------------------------
# Transformer sin/cos workaround
# ---------------------------------------------------------------------------


class WanTransformerNoSinCos(nn.Module):
    """Avoids sin/cos on device (Blackhole SFPI compiler bug).
    Pre-computed timestep embedding is passed in instead of raw timestep."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep_emb, encoder_hidden_states):
        t = self.transformer
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = t.config.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        rotary_emb = t.rope(hidden_states)
        hidden_states = t.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        ce = t.condition_embedder
        dt = next(iter(ce.time_embedder.parameters())).dtype
        if timestep_emb.dtype != dt and dt != torch.int8:
            timestep_emb = timestep_emb.to(dt)
        temb = ce.time_embedder(timestep_emb).type_as(encoder_hidden_states)
        timestep_proj = ce.time_proj(ce.act_fn(temb))
        encoder_hidden_states = ce.text_embedder(encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        for block in t.blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )

        shift, scale = (t.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(
            2, dim=1
        )
        hidden_states = (
            t.norm_out(hidden_states.float()) * (1 + scale.to(hidden_states.device))
            + shift.to(hidden_states.device)
        ).type_as(hidden_states)
        hidden_states = t.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, ppf, pph, ppw, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


# ---------------------------------------------------------------------------
# Stage 1: Text encoding
# ---------------------------------------------------------------------------


def stage_encode(prompt, max_length):
    import torch_xla
    import torch_xla.runtime as xr
    from transformers import AutoTokenizer, UMT5EncoderModel

    print("=" * 70)
    print("STAGE 1: TEXT ENCODER (UMT5-XXL)")
    print("=" * 70)

    xr.set_device_type("TT")
    device = torch_xla.device()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float32
    )
    text_encoder.eval()
    print(
        f"[load] {sum(p.numel() for p in text_encoder.parameters()) / 1e9:.2f}B params in {time.time() - t0:.1f}s"
    )

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    print(f"[input] '{prompt}' → {text_inputs.input_ids.shape}")

    text_encoder = text_encoder.to(device)
    compiled = torch.compile(text_encoder, backend="tt")

    t0 = time.time()
    with torch.no_grad():
        output = compiled(
            input_ids=text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )
    torch_xla.sync(wait=True)
    print(f"[run] {time.time() - t0:.1f}s")

    result = output.last_hidden_state.cpu()
    print(f"[result] {result.shape}, NaN: {result.isnan().any().item()}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    torch.save(result, ENC_PATH)
    print(f"[save] {ENC_PATH}")


# ---------------------------------------------------------------------------
# Stage 2: Denoising loop
# ---------------------------------------------------------------------------


def stage_denoise(num_frames, height, width, num_inference_steps, seed):
    import torch_xla
    import torch_xla.runtime as xr
    from diffusers import UniPCMultistepScheduler, WanTransformer3DModel
    from diffusers.models.embeddings import get_timestep_embedding

    print("=" * 70)
    print("STAGE 2: TRANSFORMER DENOISING LOOP")
    print("=" * 70)

    enc = torch.load(ENC_PATH, weights_only=True)
    print(f"[load] encoder_hidden_states: {enc.shape}")

    xr.set_device_type("TT")
    device = torch_xla.device()

    t0 = time.time()
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.float32
    )
    transformer.eval()
    freq_dim = transformer.config.freq_dim
    print(
        f"[load] {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B params in {time.time() - t0:.1f}s"
    )

    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    nlf = (num_frames - 1) // VAE_SCALE_TEMPORAL + 1
    lh, lw = height // VAE_SCALE_SPATIAL, width // VAE_SCALE_SPATIAL
    print(f"[input] latents: [1, {LATENT_CHANNELS}, {nlf}, {lh}, {lw}]")

    latents = torch.randn(
        1,
        LATENT_CHANNELS,
        nlf,
        lh,
        lw,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float32,
    )

    wrapper = WanTransformerNoSinCos(transformer)
    wrapper.eval()
    wrapper = wrapper.to(device)
    compiled = torch.compile(wrapper, backend="tt")

    scheduler.set_timesteps(num_inference_steps)
    enc_dev = enc.to(device)

    t0 = time.time()
    for i, t_val in enumerate(scheduler.timesteps):
        st = time.time()
        t_tensor = t_val.unsqueeze(0) if t_val.dim() == 0 else t_val
        te = get_timestep_embedding(
            t_tensor.cpu(), freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )

        with torch.no_grad():
            pred = compiled(latents.to(device), te.to(device), enc_dev)
        torch_xla.sync(wait=True)

        latents = scheduler.step(pred.cpu(), t_val, latents, return_dict=False)[0]
        print(
            f"  step {i+1}/{num_inference_steps} (t={t_val.item():.1f}) — {time.time()-st:.1f}s"
        )

    elapsed = time.time() - t0
    print(f"[run] {elapsed:.1f}s total ({elapsed/num_inference_steps:.1f}s/step)")
    print(f"[result] {latents.shape}, NaN: {latents.isnan().any().item()}")

    torch.save(latents, LATENTS_PATH)
    print(f"[save] {LATENTS_PATH}")


# ---------------------------------------------------------------------------
# Stage 3: VAE decode
# ---------------------------------------------------------------------------


def stage_decode():
    import torch_xla
    import torch_xla.runtime as xr
    from diffusers import AutoencoderKLWan

    print("=" * 70)
    print("STAGE 3: VAE DECODE")
    print("=" * 70)

    latents = torch.load(LATENTS_PATH, weights_only=True)
    print(f"[load] latents: {latents.shape}")

    xr.set_device_type("TT")
    device = torch_xla.device()

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )
    vae.eval()
    print(f"[load] VAE: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M params")

    lm = torch.tensor(vae.config.latents_mean).view(1, LATENT_CHANNELS, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std).view(1, LATENT_CHANNELS, 1, 1, 1)
    latents = latents / ls + lm

    decoder = vae.decoder.to(device)
    compiled = torch.compile(decoder, backend="tt")

    t0 = time.time()
    with torch.no_grad():
        video = compiled(latents.to(device))
    torch_xla.sync(wait=True)
    print(f"[run] {time.time() - t0:.1f}s")

    if isinstance(video, tuple):
        video = video[0]
    video_cpu = video.cpu()
    print(f"[result] {video_cpu.shape}, NaN: {video_cpu.isnan().any().item()}")

    torch.save(video_cpu, VIDEO_PATH)
    print(f"[save] {VIDEO_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Wan 2.1 T2V — isolated stages")
    parser.add_argument(
        "--stage", required=True, choices=["encode", "denoise", "decode", "all"]
    )
    parser.add_argument(
        "--prompt", default="A cat sitting on a windowsill watching rain"
    )
    parser.add_argument("--output", default="wan21_output.mp4")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--small", action="store_true", help="Use test-size dims (64x64)."
    )
    args = parser.parse_args()

    if args.small:
        args.height = SMALL_HEIGHT
        args.width = SMALL_WIDTH
    max_length = 32 if args.small else 226

    assert (
        args.num_frames - 1
    ) % 4 == 0, f"num_frames must be 1 + 4*N (got {args.num_frames})"

    if args.stage == "all":
        import sys

        base = f"{sys.executable} {__file__}"
        flags = f"--num_frames {args.num_frames} --height {args.height} --width {args.width}"
        flags += f" --num_inference_steps {args.num_inference_steps} --seed {args.seed}"
        if args.small:
            flags += " --small"

        total_t0 = time.time()
        for stage in ["encode", "denoise", "decode"]:
            extra = f' --prompt "{args.prompt}"' if stage == "encode" else ""
            cmd = f"{base} --stage {stage} {flags}{extra}"
            print(f"\n>>> {cmd}\n")
            rc = os.system(cmd)
            if rc != 0:
                print(f"Stage '{stage}' failed (exit {rc})")
                return

        if os.path.exists(VIDEO_PATH):
            video = torch.load(VIDEO_PATH, weights_only=True)
            video = (video.clamp(-1, 1) + 1) / 2
            video = (video[0].permute(1, 2, 3, 0).numpy() * 255).astype("uint8")
            from pathlib import Path

            out = str(Path(args.output).with_suffix(".mp4"))
            try:
                import PIL.Image
                from diffusers.utils import export_to_video

                frames = [PIL.Image.fromarray(video[i]) for i in range(video.shape[0])]
                export_to_video(frames, out, fps=DEFAULT_FPS)
            except ImportError:
                import imageio

                imageio.mimwrite(out, video, fps=DEFAULT_FPS)
            print(f"\n[output] {video.shape[0]} frames → {out}")

        print(f"[total] {time.time() - total_t0:.1f}s")
        print("DONE")
        return

    if args.stage == "encode":
        stage_encode(args.prompt, max_length)
    elif args.stage == "denoise":
        stage_denoise(
            args.num_frames,
            args.height,
            args.width,
            args.num_inference_steps,
            args.seed,
        )
    elif args.stage == "decode":
        stage_decode()


if __name__ == "__main__":
    main()
