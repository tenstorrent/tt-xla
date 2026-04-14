"""Debug pipeline PCC: run on CPU and verify output is not noise."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import copy


def compute_pcc(a, b, name=""):
    a_f = a.detach().float().flatten()
    b_f = b.detach().float().flatten()
    pcc = torch.corrcoef(torch.stack([a_f, b_f]))[0, 1].item()
    mx = (a_f - b_f).abs().max().item()
    print(f"  {name}: PCC={pcc:.6f}, max_diff={mx:.4f}, a_range=[{a_f.min():.3f},{a_f.max():.3f}], b_range=[{b_f.min():.3f},{b_f.max():.3f}]")
    return pcc


def run_cpu_pipeline():
    """Run a minimal pipeline entirely on CPU to check if it produces non-noise output."""
    from diffusers import LTX2VideoTransformer3DModel, AutoencoderKLLTX2Video
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    torch.manual_seed(42)
    num_frames, h, w = 2, 4, 4
    n_video = num_frames * h * w
    n_audio = 8
    latent_channels = 128
    audio_latent_channels = 8
    latent_mel_bins = 16

    # Create 4-layer transformer
    transformer = LTX2VideoTransformer3DModel(num_layers=4, rope_type="split").to(torch.bfloat16).eval()

    # Initialize latents
    video_latents = torch.randn(1, latent_channels, num_frames, h, w, dtype=torch.bfloat16)
    audio_latents = torch.randn(1, audio_latent_channels, n_audio, latent_mel_bins, dtype=torch.bfloat16)

    # Pack
    def pack_video(lat):
        B, C, F, H, W = lat.shape
        return lat.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

    def unpack_video(lat, F, H, W):
        B, S, C = lat.shape
        return lat.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)

    def pack_audio(lat):
        B, C, L, M = lat.shape
        return lat.permute(0, 2, 1, 3).reshape(B, L, C * M)

    def unpack_audio(lat, L, M):
        B, S, D = lat.shape
        C = D // M
        return lat.reshape(B, L, C, M).permute(0, 2, 1, 3)

    vl = pack_video(video_latents)
    al = pack_audio(audio_latents)

    # Simple conditioning (random, no text encoder for CPU debug)
    text_len = 16
    video_text = torch.randn(1, text_len, 3840, dtype=torch.bfloat16)
    audio_text = torch.randn(1, text_len, 3840, dtype=torch.bfloat16)

    # Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler()
    audio_scheduler = copy.deepcopy(scheduler)

    # Calculate shift
    seq_len = n_video
    mu = seq_len * ((1.16 - 0.5) / (4096 - 256)) + 0.5 - 256 * ((1.16 - 0.5) / (4096 - 256))

    num_steps = 3
    scheduler.set_timesteps(num_steps, device="cpu", mu=mu)
    audio_scheduler.set_timesteps(num_steps, device="cpu", mu=mu)

    vl = vl * scheduler.sigmas[0]
    al = al * audio_scheduler.sigmas[0]

    enc_mask = torch.ones(1, text_len, dtype=torch.long)

    print("Running CPU denoising...")
    for i, t in enumerate(scheduler.timesteps):
        t_val = t.item()
        ts = torch.tensor([t_val], dtype=torch.long)
        with torch.no_grad():
            out = transformer(
                hidden_states=vl, audio_hidden_states=al,
                encoder_hidden_states=video_text,
                audio_encoder_hidden_states=audio_text,
                timestep=ts, encoder_attention_mask=enc_mask,
                audio_encoder_attention_mask=enc_mask,
                num_frames=num_frames, height=h, width=w,
                audio_num_frames=n_audio,
            )
        noise_v = out.sample
        noise_a = out.audio_sample

        vl = scheduler.step(noise_v, t, vl).prev_sample
        al = audio_scheduler.step(noise_a, t, al).prev_sample

        print(f"  Step {i+1}: vl range=[{vl.min():.3f},{vl.max():.3f}], al range=[{al.min():.3f},{al.max():.3f}]")

    # Unpack
    video_out = unpack_video(vl, num_frames, h, w)
    print(f"\nCPU denoised video latent: shape={video_out.shape}")
    print(f"  Range: [{video_out.min():.3f}, {video_out.max():.3f}]")
    print(f"  Mean: {video_out.mean():.3f}, Std: {video_out.std():.3f}")

    # Check if it's just noise (std should change over steps)
    is_noise = video_out.std() > 5.0  # random noise has std ~1, denoised should be different
    print(f"  Looks like noise? {is_noise}")

    # Now try with diffusers pipeline on CPU for reference
    print("\n--- Diffusers LTX2Pipeline reference (CPU) ---")
    try:
        from diffusers import LTX2Pipeline
        pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
        pipe = pipe.to("cpu")
        # Just check if we can load it
        print(f"  Pipeline loaded OK. Components: {list(pipe.components.keys())}")
        print(f"  VAE latents_mean: {pipe.vae.latents_mean[:5]}")
        print(f"  VAE latents_std: {pipe.vae.latents_std[:5]}")
        print(f"  VAE scaling_factor: {pipe.vae.config.scaling_factor}")
        print(f"  Audio VAE latents_mean: {pipe.audio_vae.latents_mean[:5]}")
        print(f"  Audio VAE latents_std: {pipe.audio_vae.latents_std[:5]}")
    except Exception as e:
        print(f"  Pipeline load failed: {e}")

    return video_out


if __name__ == "__main__":
    run_cpu_pipeline()
