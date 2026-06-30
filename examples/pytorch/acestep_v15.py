# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 text-to-music diffusion pipeline example.

This mirrors the Stable-Diffusion examples (``sd_v1_4_pipeline.py``): the heavy
per-step compute -- here the ``AceStepDiTModel`` flow-matching denoiser, the key
component of the ACE-Step/Ace-Step1.5 pipeline -- runs on the Tenstorrent device while
the host drives a short scheduler loop, and the Oobleck audio VAE turns the final
acoustic latent into a 48 kHz stereo waveform.

Both models are built through the tt-forge-models loader API
(``acestep.denoiser`` and ``acestep.vae``). The denoiser is compiled with
``backend="tt"`` and run for a few flow-matching Euler steps; the VAE decode runs on
host (as in ``sd_v1_4_pipeline.py``, whose VAE defaults to CPU). The conditioning
tensors come from the denoiser loader's representative input batch -- the bringup
loaders intentionally do not expose ACE-Step's text/lyric/timbre conditioning encoder
(its custom ``trust_remote_code`` wrapper builds an FSQ quantizer that is incompatible
with ``transformers>=5`` meta-device init), so this example demonstrates the on-device
generate-and-decode pipeline mechanics rather than musically-conditioned output.
"""

from pathlib import Path

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from scipy.io import wavfile

from third_party.tt_forge_models.acestep.denoiser.pytorch import (
    ModelLoader as DenoiserLoader,
    ModelVariant as DenoiserVariant,
)
from third_party.tt_forge_models.acestep.vae.pytorch import (
    ModelLoader as VaeLoader,
    ModelVariant as VaeVariant,
)


class AceStepPipeline:
    """Text-to-music diffusion pipeline driving the ACE-Step 1.5 DiT denoiser on TT."""

    def __init__(self):
        self.denoiser_loader = DenoiserLoader(DenoiserVariant.V15_TURBO)
        self.vae_loader = VaeLoader(VaeVariant.OOBLECK)
        self.denoiser = None
        self.vae = None
        self.base_inputs = None
        self.sampling_rate = None

    def setup(self):
        # --- DiT denoiser (key compute) -> Tenstorrent ---
        self.denoiser = self.denoiser_loader.load_model()
        self.denoiser.compile(backend="tt")
        self.denoiser = self.denoiser.to(xm.xla_device())

        # A representative single-step input batch: gives valid conditioning
        # (encoder hidden states, context latents, masks) and the initial noisy
        # latent at the denoiser's native 512-frame geometry (~20.5 s of audio).
        self.base_inputs = self.denoiser_loader.load_inputs(batch_size=1)

        # --- Oobleck VAE decoder (latent -> waveform) on host ---
        self.vae = self.vae_loader.load_model()
        self.sampling_rate = self.vae.vae.config.sampling_rate

    def generate(self, num_inference_steps: int = 8, seed: int = 0) -> torch.Tensor:
        """Run the flow-matching denoise loop on TT, then VAE-decode to a waveform.

        Returns the decoded audio as a float32 tensor of shape (B, channels, samples).
        """
        device = xm.xla_device()
        base = self.base_inputs

        tt_cast = lambda x: x.to(dtype=torch.bfloat16).to(device=device)
        cpu_f32 = lambda x: x.to("cpu").to(dtype=torch.float32)

        with torch.no_grad():
            # Conditioning is fixed across the loop -> move it to device once.
            cond = {
                "attention_mask": tt_cast(base["attention_mask"]),
                "encoder_hidden_states": tt_cast(base["encoder_hidden_states"]),
                "encoder_attention_mask": tt_cast(base["encoder_attention_mask"]),
                "context_latents": tt_cast(base["context_latents"]),
            }

            # Initial noisy acoustic latent x_1 (t=1) -> data (t=0).
            torch.manual_seed(seed)
            latent = torch.randn_like(base["hidden_states"], dtype=torch.float32)
            batch = latent.shape[0]

            # Flow-matching Euler integration over a turbo-sized step schedule.
            sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
            for i in range(num_inference_steps):
                t, t_next = sigmas[i], sigmas[i + 1]
                timestep = torch.full((batch,), t.item())
                timestep_r = torch.full((batch,), t_next.item())

                outputs = self.denoiser(
                    hidden_states=tt_cast(latent),
                    timestep=tt_cast(timestep),
                    timestep_r=tt_cast(timestep_r),
                    attention_mask=cond["attention_mask"],
                    encoder_hidden_states=cond["encoder_hidden_states"],
                    encoder_attention_mask=cond["encoder_attention_mask"],
                    context_latents=cond["context_latents"],
                    use_cache=False,
                )
                # outputs[0] is the predicted flow field (velocity) over the latent.
                velocity = cpu_f32(outputs[0])
                latent = latent + (t_next - t) * velocity
                print(f"  step {i + 1}/{num_inference_steps} (t={t.item():.3f})")

            # --- VAE decode: [B, T, C] latent -> [B, C, T] -> stereo waveform ---
            vae_latent = latent.transpose(1, 2).contiguous()
            waveform = self.vae(vae_latent)

        return waveform.to(dtype=torch.float32)


def save_audio(waveform: torch.Tensor, sampling_rate: int, filepath: str) -> str:
    """Save the first batch item as a 16-bit PCM stereo WAV file."""
    audio = waveform[0].cpu().numpy()  # (channels, samples)
    audio = audio.T  # (samples, channels) for WAV
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    audio_i16 = (audio * 32767.0).astype(np.int16)
    wavfile.write(filepath, sampling_rate, audio_i16)
    return filepath


def post_process_output(waveform: torch.Tensor, sampling_rate: int, filepath: str):
    """Print a human-readable summary and save the generated audio."""
    channels, samples = waveform.shape[1], waveform.shape[2]
    duration = samples / sampling_rate
    rms = torch.sqrt(torch.mean(waveform.float() ** 2)).item()
    save_audio(waveform, sampling_rate, filepath)
    print("\nGenerated audio:")
    print(f"  channels:      {channels}")
    print(f"  samples:       {samples}")
    print(f"  sampling rate: {sampling_rate} Hz")
    print(f"  duration:      {duration:.2f} s")
    print(f"  RMS level:     {rms:.4f}")
    print(f"  saved to:      {filepath}")


def run_acestep_pipeline(
    output_path: str = "acestep_output.wav", num_inference_steps: int = 8
) -> torch.Tensor:
    """Build the ACE-Step pipeline, run it on TT, and save the audio."""
    # Match the diffusion bringup's compiler options for a fast first compile.
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    pipeline = AceStepPipeline()
    pipeline.setup()
    waveform = pipeline.generate(num_inference_steps=num_inference_steps, seed=0)
    post_process_output(waveform, pipeline.sampling_rate, output_path)
    return waveform


def test_acestep_v15():
    """ACE-Step 1.5 pipeline produces a finite, correctly-shaped 48 kHz stereo clip."""
    xr.set_device_type("TT")

    output_path = "test_acestep_output.wav"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    try:
        waveform = run_acestep_pipeline(
            output_path=output_path, num_inference_steps=8
        )

        # Oobleck VAE: 48 kHz stereo; 512 latent frames * product(downsampling) 1920.
        assert torch.isfinite(waveform).all(), "Waveform contains non-finite values"
        assert waveform.shape[1] == 2, f"Expected stereo, got {waveform.shape[1]} ch"
        assert waveform.shape[2] == 512 * 1920, f"Unexpected length {waveform.shape[2]}"
        assert output_file.exists(), f"Output audio {output_path} was not created"

        print("ACE-Step pipeline produced a valid stereo waveform.")
    finally:
        if output_file.exists():
            output_file.unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    xr.set_device_type("TT")
    run_acestep_pipeline()
