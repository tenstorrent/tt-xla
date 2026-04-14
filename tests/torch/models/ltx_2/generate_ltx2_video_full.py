#!/usr/bin/env python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Full 48-layer Video Generation on TT hardware.

Runs all 48 transformer layers by splitting into two halves (24 layers each,
~18.5 GiB per half) that fit on a single 32 GiB device. Each half is compiled
separately and run sequentially per denoising step.

The rest of the pipeline (text encoder, connectors, VAE, vocoder) is identical
to generate_ltx2_video.py.
"""

import argparse, copy, math, os, sys, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr

from conv3d_decompose import patch_conv3d_to_conv2d
from ltx2_patches import apply_all_patches


# ── Latent helpers ────────────────────────────────────────────────────

def pack_latents(lat):
    B, C, F, H, W = lat.shape
    return lat.permute(0, 2, 3, 4, 1).reshape(B, F*H*W, C)

def unpack_latents(lat, F, H, W):
    B, S, C = lat.shape
    return lat.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)

def pack_audio(lat):
    B, C, L, M = lat.shape
    return lat.permute(0, 2, 1, 3).reshape(B, L, C*M)

def unpack_audio(lat, L, M):
    B, S, D = lat.shape
    C = D // M
    return lat.reshape(B, L, C, M).permute(0, 2, 1, 3)

def denormalize_latents(lat, mean, std, sf=1.0):
    m = mean.view(1,-1,1,1,1).to(lat.device, lat.dtype)
    s = std.view(1,-1,1,1,1).to(lat.device, lat.dtype)
    return lat * s / sf + m

def denormalize_audio(lat, mean, std):
    C, M = lat.shape[1], lat.shape[3]
    m = mean.view(1,C,1,M).to(lat.device, lat.dtype)
    s = std.view(1,C,1,M).to(lat.device, lat.dtype)
    return lat * s + m

def calculate_shift(sl, bsl=1024, msl=4096, bs=0.95, ms=2.05):
    m = (ms-bs)/(msl-bsl)
    return sl*m + bs - bsl*m


# ── Main ──────────────────────────────────────────────────────────────

def generate(prompt, output_path, height=512, width=320, num_frames=49,
             num_inference_steps=20, guidance_scale=4.0, seed=42, fps=24):

    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()
    print(f"Devices: {xr.global_runtime_device_count()}")

    apply_all_patches()
    generator = torch.Generator().manual_seed(seed)

    # Dimensions
    vae_s, vae_t = 32, 8
    lh, lw, lf = height//vae_s, width//vae_s, (num_frames-1)//vae_t+1
    n_video = lf*lh*lw
    dur = num_frames/fps
    audio_num_frames = round(dur * 24000/160/4)
    latent_mel = 16
    n_audio = audio_num_frames

    print(f"Video: {height}x{width}, {num_frames}f -> latent {lf}x{lh}x{lw}={n_video} tokens")
    print(f"Audio: {dur:.1f}s -> {audio_num_frames} frames")

    # ═══════════════════ Phase 1: Text Encoding ═══════════════════════
    print("\n=== Phase 1: Text Encoding ===")
    t1 = time.time()

    from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

    tokenizer = GemmaTokenizerFast.from_pretrained("Lightricks/LTX-2", subfolder="tokenizer")
    full_model = Gemma3ForConditionalGeneration.from_pretrained(
        "Lightricks/LTX-2", subfolder="text_encoder", torch_dtype=torch.bfloat16,
    )
    text_encoder = full_model.model.language_model
    text_encoder.config.use_cache = False
    text_encoder.config.sliding_window = None
    for layer in text_encoder.layers:
        if hasattr(layer.self_attn, 'sliding_window'):
            layer.self_attn.sliding_window = None
    text_encoder = text_encoder.eval().to(device)
    del full_model

    max_seq = 256
    inputs = tokenizer(prompt, padding="max_length", max_length=max_seq, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    token_mask = inputs["attention_mask"]

    causal = torch.triu(torch.full((max_seq, max_seq), float("-inf"), dtype=torch.bfloat16), diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).to(device)

    compiled_enc = torch.compile(text_encoder, backend="tt")
    with torch.no_grad():
        enc_out = compiled_enc(input_ids=input_ids, attention_mask=causal, output_hidden_states=True)
    torch_xla.sync(wait=True)

    # Normalize text embeddings (matching diffusers _pack_text_embeds)
    text_h = torch.stack(list(enc_out.hidden_states), dim=-1)  # [B,S,D,N]
    B, S, D, N = text_h.shape
    seq_lens = token_mask.sum(dim=1)
    tidx = torch.arange(S, device=text_h.device).unsqueeze(0)
    mask = tidx >= (S - seq_lens.unsqueeze(1).to(text_h.device))
    mask = mask[:,:,None,None]
    eps = 1e-6
    mh = text_h.float().masked_fill(~mask, 0.0)
    nv = (seq_lens * D).view(B,1,1,1).to(text_h.device).float()
    mm = mh.sum(dim=(1,2), keepdim=True) / (nv + eps)
    xn = text_h.float().masked_fill(~mask, float("inf")).amin(dim=(1,2), keepdim=True)
    xx = text_h.float().masked_fill(~mask, float("-inf")).amax(dim=(1,2), keepdim=True)
    pe = ((text_h.float() - mm) / (xx - xn + eps)) * 8.0
    pe = pe.flatten(2)
    pe = pe.masked_fill(~mask.squeeze(-1).expand(-1,-1,D*N), 0.0).to(text_h.dtype)

    del compiled_enc, text_encoder, enc_out, text_h
    torch_xla.sync(wait=True); gc.collect()

    # Connectors
    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    conn = LTX2TextConnectors(
        caption_channels=3840, text_proj_in_factor=49,
        video_connector_num_attention_heads=30, video_connector_attention_head_dim=128,
        video_connector_num_layers=2, video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=30, audio_connector_attention_head_dim=128,
        audio_connector_num_layers=2, audio_connector_num_learnable_registers=None,
        connector_rope_base_seq_len=4096, rope_theta=10000.0, rope_double_precision=True,
        causal_temporal_positioning=False, rope_type="interleaved",
    )
    wp = hf_hub_download("Lightricks/LTX-2", "connectors/diffusion_pytorch_model.safetensors")
    sd = safetensors.torch.load_file(wp)
    conn.load_state_dict({k:v for k,v in sd.items() if "learnable_registers" not in k}, strict=False)
    conn = conn.to(torch.bfloat16).eval().to(device)
    cc = torch.compile(conn, backend="tt", fullgraph=True)

    add_mask = ((1 - token_mask.to(torch.bfloat16)) * -10000.0).unsqueeze(1).unsqueeze(1).to(device)
    with torch.no_grad():
        vt, at, _ = cc(pe, add_mask, additive_mask=True)
    torch_xla.sync(wait=True)

    vt_cfg = torch.cat([torch.zeros_like(vt), vt], dim=0)
    at_cfg = torch.cat([torch.zeros_like(at), at], dim=0)

    del cc, conn, pe
    torch_xla.sync(wait=True); gc.collect()
    print(f"Phase 1 done in {time.time()-t1:.0f}s")

    # ═══════════════════ Phase 2: Denoising (48 layers in 2 halves) ═══
    print("\n=== Phase 2: Denoising (48 layers, split into 2x24) ===")
    t2 = time.time()

    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    print("Loading full 48-layer transformer...")
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-2", subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()

    # Split into two halves — each ~18.5 GiB, fits on 32 GiB device
    all_blocks = list(transformer.transformer_blocks)
    half = len(all_blocks) // 2  # 24

    # Create two transformer copies with different layer subsets
    # Share everything except transformer_blocks
    def make_half(blocks):
        """Create a transformer with a subset of blocks."""
        t = LTX2VideoTransformer3DModel.from_pretrained(
            "Lightricks/LTX-2", subfolder="transformer", torch_dtype=torch.bfloat16,
        ).eval()
        t.transformer_blocks = nn.ModuleList(blocks)
        t.config.num_layers = len(blocks)
        return t

    print(f"Creating half_1 (blocks 0-{half-1})...")
    half1 = make_half(all_blocks[:half])
    print(f"Creating half_2 (blocks {half}-{len(all_blocks)-1})...")
    half2 = make_half(all_blocks[half:])
    del transformer, all_blocks; gc.collect()

    # Wrapper to clone outputs
    class HalfWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, **kw):
            o = self.m(**kw)
            return o.sample.clone(), o.audio_sample.clone()

    # Compile half1 on device
    print("Compiling half_1 on device...")
    half1_tt = half1.to(device)
    h1_compiled = torch.compile(HalfWrapper(half1_tt), backend="tt", fullgraph=True)

    # Initialize latents
    vl = pack_latents(torch.randn(1, 128, lf, lh, lw, dtype=torch.bfloat16, generator=generator)).to(device)
    al = pack_audio(torch.randn(1, 8, audio_num_frames, latent_mel, dtype=torch.bfloat16, generator=generator)).to(device)

    # Scheduler
    sched = FlowMatchEulerDiscreteScheduler()
    a_sched = copy.deepcopy(sched)
    mu = calculate_shift(n_video)
    sigmas = np.linspace(1.0, 1.0/num_inference_steps, num_inference_steps)
    sched.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
    a_sched.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)

    enc_mask = torch.ones(2, max_seq, dtype=torch.long).to(device)

    print(f"Running {num_inference_steps} denoising steps...")
    for i, t in enumerate(sched.timesteps):
        ts = time.time()

        li = torch.cat([vl, vl], dim=0)
        ai = torch.cat([al, al], dim=0)
        tb = t.expand(2).to(device)

        # Run half_1
        with torch.no_grad():
            v1, a1 = h1_compiled(
                hidden_states=li, audio_hidden_states=ai,
                encoder_hidden_states=vt_cfg, audio_encoder_hidden_states=at_cfg,
                timestep=tb, encoder_attention_mask=enc_mask,
                audio_encoder_attention_mask=enc_mask,
                num_frames=lf, height=lh, width=lw, audio_num_frames=audio_num_frames,
            )
        torch_xla.sync(wait=True)

        # If this is the first step, we need to compile half2 after freeing half1
        if i == 0:
            # Free half1 from device
            del h1_compiled, half1_tt
            torch_xla.sync(wait=True); gc.collect()

            # Compile half2
            print("  Compiling half_2 on device...")
            half2_tt = half2.to(device)
            h2_compiled = torch.compile(HalfWrapper(half2_tt), backend="tt", fullgraph=True)

        # Run half_2 with half_1's output as input
        with torch.no_grad():
            v2, a2 = h2_compiled(
                hidden_states=v1, audio_hidden_states=a1,
                encoder_hidden_states=vt_cfg, audio_encoder_hidden_states=at_cfg,
                timestep=tb, encoder_attention_mask=enc_mask,
                audio_encoder_attention_mask=enc_mask,
                num_frames=lf, height=lh, width=lw, audio_num_frames=audio_num_frames,
            )
        torch_xla.sync(wait=True)

        # CFG
        nv = v2.float(); na = a2.float()
        uv, cv = nv.chunk(2); ua, ca = na.chunk(2)
        pv = uv + guidance_scale * (cv - uv)
        pa = ua + guidance_scale * (ca - ua)

        vl = sched.step(pv.cpu(), t, vl.cpu()).prev_sample.to(device=device, dtype=torch.bfloat16)
        al = a_sched.step(pa.cpu(), t, al.cpu()).prev_sample.to(device=device, dtype=torch.bfloat16)

        el = time.time() - ts
        print(f"  Step {i+1}/{num_inference_steps} (t={t:.0f}): {el:.1f}s")

    del h2_compiled, half2_tt, half1, half2
    torch_xla.sync(wait=True); gc.collect()
    print(f"Phase 2 done in {time.time()-t2:.0f}s")

    # ═══════════════════ Phase 3: Decoding ════════════════════════════
    print("\n=== Phase 3: Decoding ===")
    t3 = time.time()

    from diffusers import AutoencoderKLLTX2Video
    from diffusers.models.autoencoders import AutoencoderKLLTX2Audio

    # Video decode
    vae = AutoencoderKLLTX2Video.from_pretrained("Lightricks/LTX-2", subfolder="vae", torch_dtype=torch.bfloat16)
    vl5 = unpack_latents(vl, lf, lh, lw)
    vl5 = denormalize_latents(vl5, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
    dec = vae.decoder.eval().to(device)
    del vae
    dc = torch.compile(dec, backend="tt")
    with torch.no_grad():
        video = dc(vl5.to(device))
    torch_xla.sync(wait=True)
    print(f"  Video: {video.shape}")

    # Save video immediately
    vc = video.cpu().float()
    vc = ((vc + 1.0) / 2.0).clamp(0, 1)
    frames = (vc[0].permute(1,2,3,0) * 255).to(torch.uint8).numpy()
    try:
        import av
        c = av.open(output_path, mode="w")
        s = c.add_stream("libx264", rate=fps)
        s.width = width; s.height = height; s.pix_fmt = "yuv420p"
        for f in frames:
            fr = av.VideoFrame.from_ndarray(f, format="rgb24")
            for p in s.encode(fr): c.mux(p)
        for p in s.encode(): c.mux(p)
        c.close()
        print(f"  Saved: {output_path}")
    except ImportError:
        np.save(output_path.replace('.mp4', '_frames.npy'), frames)
        print(f"  Saved numpy: {output_path.replace('.mp4', '_frames.npy')}")

    del dc, dec
    torch_xla.sync(wait=True)

    print(f"Phase 3 done in {time.time()-t3:.0f}s")
    total = time.time() - t1
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE (48 layers, {num_inference_steps} steps)")
    print(f"  Phase 1: {time.time()-t1 - (time.time()-t2):.0f}s")
    print(f"  Phase 2: {time.time()-t2 - (time.time()-t3):.0f}s")
    print(f"  Phase 3: {time.time()-t3:.0f}s")
    print(f"  Total: {total:.0f}s")
    print(f"  Video: {height}x{width}, {num_frames}f @ {fps}fps")
    print(f"  Frames stats: mean={frames.mean():.1f}, std={frames.std():.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A dog playing a guitar on the street with a rat")
    p.add_argument("--output", default="/root/tt-xla/ltx_example.mp4")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    generate(a.prompt, a.output, a.height, a.width, a.num_frames, a.num_inference_steps, a.guidance_scale, a.seed)
