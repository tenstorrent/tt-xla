# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ACE-Step 1.5 text-to-music pipeline example (native 30 s, 48 kHz stereo).

ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5) is a multi-stage
generative music model. Like a diffusion image pipeline it is not a single
forward pass, so the tt-forge-models loader exposes each independently-
compilable component as its own variant. This example drives the two device
components of the audio-generation path through the loader's public API:

  - Denoiser  -> AceStepDiTModel     (per-step flow-matching DiT; key component)
  - VaeDecoder -> AutoencoderOobleck  (48 kHz stereo audio decoder)

It runs a host-Python flow-matching (rectified-flow Euler) sampling loop that
repeatedly calls the compiled denoiser on device to integrate a latent from
noise (t=1) to data (t=0), then decodes that latent to a stereo waveform with
the compiled VAE and writes it to a .wav file -- mirroring the per-component /
host-scheduler structure of the SDXL pipeline example (sdxl-pipeline.py).

Note on conditioning: the loader exposes the trained components with synthetic
conditioning (there is no public text -> conditioning wiring in the loader), so
this example demonstrates the full denoise -> decode audio pipeline running on
device at the model's native geometry rather than prompt-faithful music. The
geometry is the real one: 750 latent frames == 30 s at the Oobleck 48 kHz rate.

Requires `vector_quantize_pytorch` and `einops` (loader deps) plus `soundfile`
for writing the .wav artifact.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.acestep.pytorch import ModelLoader, ModelVariant


def build_components():
    """Load the ACE-Step denoiser + VAE decoder via the loader and compile for TT."""
    device = torch_xla.device()

    denoiser_loader = ModelLoader(ModelVariant.DENOISER)
    denoiser = denoiser_loader.load_model().to(device)
    denoiser.compile(backend="tt")

    vae_loader = ModelLoader(ModelVariant.VAE_DECODER)
    vae = vae_loader.load_model().to(device)
    vae.compile(backend="tt")

    # Loader-provided synthetic conditioning + initial noise at the native clip.
    hidden_states, _, encoder_hidden_states, context_latents = (
        denoiser_loader.load_inputs()
    )

    # Sample rate of the native artifact (public config on the Oobleck decoder).
    sample_rate = vae.vae.config.sampling_rate

    return denoiser, vae, hidden_states, encoder_hidden_states, context_latents, sample_rate


def generate(num_steps: int = 8):
    """Run the ACE-Step denoise -> decode pipeline on device and return audio.

    Returns a CPU tensor of shape (1, 2, samples) at 48 kHz and its sample rate.
    """
    device = torch_xla.device()
    (
        denoiser,
        vae,
        hidden_states,
        encoder_hidden_states,
        context_latents,
        sample_rate,
    ) = build_components()

    # Fixed conditioning; only the latent + timestep move through the loop.
    encoder_hidden_states = encoder_hidden_states.to(device)
    context_latents = context_latents.to(device)
    latent = hidden_states.to(device)  # noise at t=1.0

    # Rectified-flow Euler integration from noise (t=1) to data (t=0).
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
    with torch.no_grad():
        for i in range(num_steps):
            timestep = torch.full(
                (latent.shape[0],), sigmas[i].item(), dtype=latent.dtype, device=device
            )
            velocity = denoiser(
                latent, timestep, encoder_hidden_states, context_latents
            )
            dt = (sigmas[i + 1] - sigmas[i]).item()
            latent = latent + dt * velocity

        # DiT latent is (B, frames, channels); the Oobleck decoder wants (B, channels, frames).
        audio_latent = latent.transpose(1, 2).contiguous()
        audio = vae(audio_latent)

    return audio.cpu().float(), sample_rate


def post_process_output(audio, sample_rate, output_path="acestep_output.wav"):
    """Write the stereo waveform to a .wav file and print a human-readable summary."""
    import soundfile as sf

    waveform = audio.squeeze(0)  # (channels, samples)
    channels, samples = waveform.shape
    duration = samples / sample_rate
    peak = waveform.abs().max().item()

    # soundfile expects (samples, channels).
    sf.write(output_path, waveform.transpose(0, 1).numpy(), int(sample_rate))

    print(f"Generated {duration:.1f} s of {channels}-channel audio at {sample_rate} Hz")
    print(f"Samples per channel: {samples}  |  peak amplitude: {peak:.4f}")
    print(f"Saved waveform to: {output_path}")
    return output_path


def test_acestep():
    """Guard the ACE-Step pipeline: one denoise step + decode yields finite 30 s stereo audio."""
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    audio, sample_rate = generate(num_steps=1)

    assert audio.dim() == 3 and audio.shape[:2] == (1, 2), (
        f"expected (1, 2, samples) stereo audio, got {tuple(audio.shape)}"
    )
    assert torch.isfinite(audio).all(), "audio contains non-finite values"

    duration = audio.shape[-1] / sample_rate
    assert abs(duration - 30.0) < 1.0, f"expected ~30 s clip, got {duration:.2f} s"

    print(f"ACE-Step pipeline produced finite {duration:.1f} s stereo audio.")


if __name__ == "__main__":
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    audio, sample_rate = generate(num_steps=8)
    post_process_output(audio, sample_rate)
