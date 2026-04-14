# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Video Generation — Real text-to-video generation on TT hardware.

Generates a video+audio from a text prompt using:
  Phase 1: Gemma3 text encoder (pretrained) + Text Connectors -> conditioning
  Phase 2: Transformer (pretrained, 4-way SPMD TP) -> denoising loop
  Phase 3: Video VAE Decoder + Audio VAE Decoder + Vocoder -> mp4

Usage:
  python generate_ltx2_video.py --prompt "A dog playing guitar" --output ltx_example.mp4
"""

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from conv3d_decompose import patch_conv3d_to_conv2d
from ltx2_patches import apply_all_patches


# ---------------------------------------------------------------------------
# Latent helpers (from LTX2Pipeline)
# ---------------------------------------------------------------------------

def pack_latents(latents, patch_size=1, patch_size_t=1):
    """[B, C, F, H, W] -> [B, S, D] where S=F*H*W, D=C"""
    B, C, F, H, W = latents.shape
    latents = latents.reshape(B, C, F // patch_size_t, patch_size_t, H // patch_size, patch_size, W // patch_size, patch_size)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    latents = latents.flatten(4, 7).flatten(1, 3)
    return latents


def unpack_latents(latents, F, H, W, patch_size=1, patch_size_t=1):
    """[B, S, D] -> [B, C, F, H, W]"""
    B, S, D = latents.shape
    C = D // (patch_size_t * patch_size * patch_size)
    latents = latents.reshape(B, F // patch_size_t, H // patch_size, W // patch_size, C, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = latents.reshape(B, C, F, H, W)
    return latents


def pack_audio_latents(latents):
    """[B, C, L, M] -> [B, L, C*M]"""
    B, C, L, M = latents.shape
    return latents.permute(0, 2, 1, 3).reshape(B, L, C * M)


def unpack_audio_latents(latents, L, M):
    """[B, S, D] -> [B, C, L, M]"""
    B, S, D = latents.shape
    C = D // M
    return latents.reshape(B, L, C, M).permute(0, 2, 1, 3)


def normalize_latents(latents, mean, std, scaling_factor=1.0):
    mean = mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - mean) * scaling_factor / std


def denormalize_latents(latents, mean, std, scaling_factor=1.0):
    mean = mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * std / scaling_factor + mean


def denormalize_audio_latents(latents, mean, std):
    # Audio latents: [B, C, L, M]. mean/std has C*M elements (e.g. 8*16=128).
    C, M = latents.shape[1], latents.shape[3]
    mean = mean.view(1, C, 1, M).to(latents.device, latents.dtype)
    std = std.view(1, C, 1, M).to(latents.device, latents.dtype)
    return latents * std + mean


# ---------------------------------------------------------------------------
# Scheduler helper
# ---------------------------------------------------------------------------

def calculate_shift(seq_len, base_seq_len=1024, max_seq_len=4096, base_shift=0.95, max_shift=2.05):
    """Match diffusers scheduler config defaults for LTX-2."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = seq_len * m + b
    return mu


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(prompt, output_path, height=512, width=320, num_frames=49,
             num_inference_steps=20, guidance_scale=4.0, seed=42, fps=24):

    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    # Enable SPMD for tensor-parallel transformer
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    device = torch_xla.device()
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, (1, num_devices), ("batch", "model"))
    print(f"Devices: {num_devices} ({num_devices}-way TP for transformer)")

    # Apply all patches (Conv3d decompose, attention reshape, view_of clone)
    apply_all_patches()

    generator = torch.Generator().manual_seed(seed)

    # Compute latent dimensions
    vae_spatial = 32
    vae_temporal = 8
    latent_h = height // vae_spatial
    latent_w = width // vae_spatial
    latent_f = (num_frames - 1) // vae_temporal + 1
    n_video = latent_f * latent_h * latent_w
    latent_channels = 128

    # Audio dimensions
    audio_sampling_rate = 24000
    audio_hop_length = 160
    audio_vae_temporal_compression = 4
    audio_vae_mel_compression = 4
    duration_s = num_frames / fps
    audio_latents_per_second = audio_sampling_rate / audio_hop_length / audio_vae_temporal_compression
    audio_num_frames = round(duration_s * audio_latents_per_second)
    latent_mel_bins = 64 // audio_vae_mel_compression  # 16
    n_audio = audio_num_frames
    audio_latent_channels = 8

    print(f"Video: {height}x{width}, {num_frames} frames -> latent {latent_f}x{latent_h}x{latent_w} = {n_video} tokens")
    print(f"Audio: {duration_s:.1f}s -> {audio_num_frames} latent frames, {n_audio} tokens")

    # ==================================================================
    # Phase 1: Text Encoding
    # ==================================================================
    print("\n=== Phase 1: Text Encoding ===")

    from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

    t_phase1 = time.time()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GemmaTokenizerFast.from_pretrained("Lightricks/LTX-2", subfolder="tokenizer")

    # Load Gemma3 text encoder — extract the inner language model to bypass
    # the ForConditionalGeneration mask generation which uses operations
    # (sliding window slicing, padding_mask.all()) incompatible with TT.
    print("Loading Gemma3 text encoder (extracting language model)...")
    full_model = Gemma3ForConditionalGeneration.from_pretrained(
        "Lightricks/LTX-2", subfolder="text_encoder", torch_dtype=torch.bfloat16,
    )
    text_encoder = full_model.model.language_model  # Gemma3TextModel
    text_encoder.config.use_cache = False
    text_encoder.config.sliding_window = None
    for layer in text_encoder.layers:
        if hasattr(layer.self_attn, 'sliding_window'):
            layer.self_attn.sliding_window = None
    text_encoder = text_encoder.eval().to(device)
    del full_model

    # Tokenize
    max_seq_len = 256
    inputs = tokenizer(prompt, padding="max_length", max_length=max_seq_len,
                       truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    token_attention_mask = inputs["attention_mask"]  # keep on CPU for additive mask

    # Pre-compute combined causal + padding mask.
    # Must match what Gemma3ForConditionalGeneration generates internally:
    # - Causal: position q can't attend to position p > q
    # - Padding: no position can attend to padding tokens
    causal_mask = torch.triu(
        torch.full((max_seq_len, max_seq_len), float("-inf"), dtype=torch.bfloat16), diagonal=1
    )
    # Add padding mask: mask out columns where token_attention_mask is 0
    pad_mask = (1 - token_attention_mask.to(torch.bfloat16)) * float("-inf")  # [1, seq]
    pad_mask = pad_mask.unsqueeze(1).expand(-1, max_seq_len, -1)  # [1, seq, seq]
    combined_mask = causal_mask.unsqueeze(0) + pad_mask  # broadcast: [1, seq, seq]
    combined_mask = combined_mask.unsqueeze(1).to(device)  # [1, 1, seq, seq]

    print(f"Running text encoder on prompt: '{prompt}'...")
    with torch.no_grad():
        enc_out = text_encoder(input_ids=input_ids, attention_mask=combined_mask,
                               output_hidden_states=True)
    torch_xla.sync(wait=True)

    # Stack hidden states and apply masked normalization (matching diffusers _pack_text_embeds)
    num_hidden = len(enc_out.hidden_states)
    print(f"  Got {num_hidden} hidden states")
    # [B, seq, hidden, num_layers]
    text_hidden = torch.stack(list(enc_out.hidden_states), dim=-1)

    # Compute sequence lengths from attention mask
    seq_lengths = token_attention_mask.sum(dim=1)  # [B]

    # _pack_text_embeds: masked normalization per-batch, per-layer
    B, S, D, N = text_hidden.shape
    token_indices = torch.arange(S, device=text_hidden.device).unsqueeze(0)
    # Left padding (Gemma tokenizer default)
    start_indices = S - seq_lengths.unsqueeze(1).to(text_hidden.device)
    mask = token_indices >= start_indices  # [B, S]
    mask = mask[:, :, None, None]  # [B, S, 1, 1]

    eps = 1e-6
    masked_h = text_hidden.masked_fill(~mask, 0.0)
    num_valid = (seq_lengths * D).view(B, 1, 1, 1).to(text_hidden.device).float()
    masked_mean = masked_h.float().sum(dim=(1, 2), keepdim=True) / (num_valid + eps)
    x_min = text_hidden.float().masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden.float().masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    prompt_embeds = (text_hidden.float() - masked_mean) / (x_max - x_min + eps)
    prompt_embeds = prompt_embeds * 8.0  # scale_factor=8 (diffusers default)
    prompt_embeds = prompt_embeds.flatten(2)  # [B, S, D*N]
    mask_flat = mask.squeeze(-1).expand(-1, -1, D * N)
    prompt_embeds = prompt_embeds.masked_fill(~mask_flat, 0.0)
    prompt_embeds = prompt_embeds.to(text_hidden.dtype)
    print(f"  Text encoder output (normalized): {prompt_embeds.shape}")

    # Create additive attention mask from token mask
    additive_mask = (1 - token_attention_mask.to(torch.bfloat16)) * -10000.0
    additive_mask = additive_mask.unsqueeze(1).unsqueeze(1).to(device)  # [1, 1, 1, seq]

    del text_encoder, enc_out, text_hidden
    torch_xla.sync(wait=True)

    # Text Connectors (pretrained)
    print("Loading text connectors...")
    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    connectors = LTX2TextConnectors(
        caption_channels=3840, text_proj_in_factor=49,
        video_connector_num_attention_heads=30, video_connector_attention_head_dim=128,
        video_connector_num_layers=2, video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=30, audio_connector_attention_head_dim=128,
        audio_connector_num_layers=2, audio_connector_num_learnable_registers=None,
        connector_rope_base_seq_len=4096, rope_theta=10000.0, rope_double_precision=True,
        causal_temporal_positioning=False, rope_type="interleaved",
    )
    weights_path = hf_hub_download("Lightricks/LTX-2", "connectors/diffusion_pytorch_model.safetensors")
    state_dict = safetensors.torch.load_file(weights_path)
    filtered = {k: v for k, v in state_dict.items() if "learnable_registers" not in k}
    connectors.load_state_dict(filtered, strict=False)
    connectors = connectors.to(torch.bfloat16).eval().to(device)
    connectors_compiled = torch.compile(connectors, backend="tt", fullgraph=True)

    print("Running text connectors...")
    with torch.no_grad():
        video_text, audio_text, connector_mask = connectors_compiled(
            prompt_embeds, additive_mask, additive_mask=True,
        )
    torch_xla.sync(wait=True)
    print(f"  Video conditioning: {video_text.shape}, Audio conditioning: {audio_text.shape}")

    # For CFG: create unconditional embeddings (zeros)
    uncond_video_text = torch.zeros_like(video_text)
    uncond_audio_text = torch.zeros_like(audio_text)

    # Concat for CFG: [uncond, cond] along batch
    video_text_cfg = torch.cat([uncond_video_text, video_text], dim=0)
    audio_text_cfg = torch.cat([uncond_audio_text, audio_text], dim=0)

    del connectors_compiled, connectors, prompt_embeds
    torch_xla.sync(wait=True)

    t_phase1_end = time.time()
    print(f"Phase 1 done in {t_phase1_end - t_phase1:.1f}s")

    # ==================================================================
    # Phase 2: Denoising
    # ==================================================================
    print("\n=== Phase 2: Denoising ===")

    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    t_phase2 = time.time()

    # Load transformer with 4-way TP sharding.
    # Using 24 layers due to host-side compiler memory limits during compilation.
    # 48 layers OOMs the compiler; 24 layers compiles successfully.
    # TODO: Re-enable 48 layers when tt-mlir compiler memory is optimized.
    num_transformer_layers = 24
    print(f"Loading transformer ({num_transformer_layers}/48 layers, {num_devices}-way TP)...")
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-2", subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    import torch.nn as nn
    transformer.transformer_blocks = nn.ModuleList(
        list(transformer.transformer_blocks)[:num_transformer_layers]
    )
    transformer.config.num_layers = num_transformer_layers
    # Keep original "split" rope_type — our patched attention processor
    # uses the original apply_split_rotary_emb which preserves correctness.
    transformer = transformer.eval().to(device)

    # Apply Megatron-style TP sharding to all attention + FFN weights
    from run_ltx2_transformer import shard_transformer
    shard_transformer(transformer, mesh)

    class TransformerWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, **kwargs):
            out = self.inner(**kwargs)
            return out.sample.clone(), out.audio_sample.clone()

    wrapper = TransformerWrapper(transformer)
    transformer_compiled = torch.compile(wrapper, backend="tt")

    # Initialize latents
    video_latents = torch.randn(1, latent_channels, latent_f, latent_h, latent_w,
                                dtype=torch.bfloat16, generator=generator)
    audio_latents = torch.randn(1, audio_latent_channels, audio_num_frames, latent_mel_bins,
                                dtype=torch.bfloat16, generator=generator)

    # Pack latents
    video_latents_packed = pack_latents(video_latents).to(device)
    audio_latents_packed = pack_audio_latents(audio_latents).to(device)

    # Setup scheduler
    scheduler = FlowMatchEulerDiscreteScheduler()
    audio_scheduler = copy.deepcopy(scheduler)

    mu = calculate_shift(n_video)
    # Match diffusers: use linspace sigmas
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
    audio_scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
    timesteps = scheduler.timesteps
    # NOTE: diffusers does NOT scale initial latents by sigmas[0]

    # Attention masks for text conditioning
    enc_mask = torch.ones(2, max_seq_len, dtype=torch.long).to(device)  # batch=2 for CFG

    print(f"Running {num_inference_steps} denoising steps...")
    for i, t in enumerate(timesteps):
        t_step = time.time()

        # CFG: duplicate latents
        latent_input = torch.cat([video_latents_packed, video_latents_packed], dim=0)
        audio_input = torch.cat([audio_latents_packed, audio_latents_packed], dim=0)
        # Timestep must be float (not long) and expanded to batch size
        timestep_batch = t.expand(2).to(device)  # batch=2 for CFG

        with torch.no_grad():
            noise_pred_video, noise_pred_audio = transformer_compiled(
                hidden_states=latent_input,
                audio_hidden_states=audio_input,
                encoder_hidden_states=video_text_cfg,
                audio_encoder_hidden_states=audio_text_cfg,
                timestep=timestep_batch,
                encoder_attention_mask=enc_mask,
                audio_encoder_attention_mask=enc_mask,
                num_frames=latent_f,
                height=latent_h,
                width=latent_w,
                audio_num_frames=audio_num_frames,
            )
        torch_xla.sync(wait=True)

        # Apply CFG (in float, matching diffusers)
        noise_pred_video = noise_pred_video.float()
        noise_pred_audio = noise_pred_audio.float()
        uncond_v, cond_v = noise_pred_video.chunk(2, dim=0)
        uncond_a, cond_a = noise_pred_audio.chunk(2, dim=0)
        noise_pred_v = uncond_v + guidance_scale * (cond_v - uncond_v)
        noise_pred_a = uncond_a + guidance_scale * (cond_a - uncond_a)

        # Scheduler step (on CPU, then cast back to bf16 for next step)
        video_latents_packed = scheduler.step(
            noise_pred_v.cpu(), t, video_latents_packed.cpu()
        ).prev_sample.to(device=device, dtype=torch.bfloat16)
        audio_latents_packed = audio_scheduler.step(
            noise_pred_a.cpu(), t, audio_latents_packed.cpu()
        ).prev_sample.to(device=device, dtype=torch.bfloat16)

        elapsed = time.time() - t_step
        print(f"  Step {i+1}/{num_inference_steps} (t={t:.1f}): {elapsed:.1f}s")

    del transformer_compiled, wrapper, transformer
    torch_xla.sync(wait=True)

    t_phase2_end = time.time()
    print(f"Phase 2 done in {t_phase2_end - t_phase2:.1f}s")

    # ==================================================================
    # Phase 3: Decoding
    # ==================================================================
    print("\n=== Phase 3: Decoding ===")
    t_phase3 = time.time()

    from diffusers import AutoencoderKLLTX2Video
    from diffusers.models.autoencoders import AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2 import LTX2Vocoder

    # Unpack video latents
    video_latents_5d = unpack_latents(video_latents_packed, latent_f, latent_h, latent_w)

    # Denormalize video latents
    vae = AutoencoderKLLTX2Video.from_pretrained(
        "Lightricks/LTX-2", subfolder="vae", torch_dtype=torch.bfloat16,
    )
    video_latents_5d = denormalize_latents(
        video_latents_5d, vae.latents_mean, vae.latents_std, vae.config.scaling_factor,
    )
    video_decoder = vae.decoder.eval().to(device)
    del vae

    video_decoder_compiled = torch.compile(video_decoder, backend="tt")

    print("Decoding video...")
    with torch.no_grad():
        video = video_decoder_compiled(video_latents_5d.to(device))
    torch_xla.sync(wait=True)
    print(f"  Video shape: {video.shape}")

    del video_decoder_compiled, video_decoder
    torch_xla.sync(wait=True)

    # Audio decode
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
        "Lightricks/LTX-2", subfolder="audio_vae", torch_dtype=torch.bfloat16,
    )
    audio_latents_4d = unpack_audio_latents(audio_latents_packed, audio_num_frames, latent_mel_bins)
    audio_latents_4d = denormalize_audio_latents(
        audio_latents_4d, audio_vae.latents_mean, audio_vae.latents_std,
    )
    audio_decoder = audio_vae.decoder.eval().to(device)
    del audio_vae

    audio_decoder_compiled = torch.compile(audio_decoder, backend="tt")

    print("Decoding audio mel spectrogram...")
    with torch.no_grad():
        mel = audio_decoder_compiled(audio_latents_4d.to(device))
    torch_xla.sync(wait=True)
    print(f"  Mel shape: {mel.shape}")

    del audio_decoder_compiled, audio_decoder
    torch_xla.sync(wait=True)

    # Save video IMMEDIATELY before attempting vocoder (which may crash)
    print("  Saving video (before vocoder)...")
    video_cpu = video.cpu().float()
    video_cpu = (video_cpu + 1.0) / 2.0
    video_cpu = video_cpu.clamp(0, 1)
    frames = (video_cpu[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).numpy()
    try:
        import av
        container = av.open(output_path, mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        print(f"  Video saved: {output_path} ({len(frames)} frames)")
    except ImportError:
        np.save(output_path.replace('.mp4', '_frames.npy'), frames)
        print(f"  Saved as numpy (no PyAV)")

    # Vocoder (may fail due to conv_transpose2d tt-metal bug)
    vocoder = LTX2Vocoder.from_pretrained(
        "Lightricks/LTX-2", subfolder="vocoder", torch_dtype=torch.bfloat16,
    ).eval().to(device)
    vocoder_compiled = torch.compile(vocoder, backend="tt")

    print("Running vocoder...")
    try:
        with torch.no_grad():
            audio_waveform = vocoder_compiled(mel)
        torch_xla.sync(wait=True)
        print(f"  Audio waveform: {audio_waveform.shape}")
    except Exception as e:
        print(f"  Vocoder FAILED (conv_transpose2d tt-metal bug): {type(e).__name__}")
        print(f"  Saving video without audio.")
        audio_waveform = None

    del vocoder_compiled, vocoder
    torch_xla.sync(wait=True)

    t_phase3_end = time.time()
    print(f"Phase 3 done in {t_phase3_end - t_phase3:.1f}s")

    # Video already saved above

    # ==================================================================
    # Timing summary
    # ==================================================================
    total = t_phase3_end - t_phase1
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"  Phase 1 (Text Encoding):  {t_phase1_end - t_phase1:.1f}s")
    print(f"  Phase 2 (Denoising):      {t_phase2_end - t_phase2:.1f}s")
    print(f"  Phase 3 (Decoding):       {t_phase3_end - t_phase3:.1f}s")
    print(f"  Total e2e:                {total:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Video: {height}x{width}, {num_frames} frames @ {fps}fps")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A dog playing a guitar on the street with a rat")
    parser.add_argument("--output", type=str, default="ltx_example.mp4")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    generate(
        prompt=args.prompt, output_path=args.output,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        seed=args.seed, fps=args.fps,
    )
