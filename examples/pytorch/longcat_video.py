# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LongCat-Video text-to-video pipeline example.

LongCat-Video (meituan-longcat/LongCat-Video) is a 13.6B text-to-video diffusion
pipeline built from three components, each brought up by its own tt-forge-models
loader:

  * a UMT5-XXL text encoder that embeds the prompt,
  * a custom 13.6B 3D diffusion transformer (`LongCatVideoTransformer3DModel`,
    depth 48, hidden 4096, patch [1,2,2], 16 latent channels) that denoises the
    video latent under cross-attention conditioning,
  * a Wan VAE decoder that turns the denoised latent into RGB frames.

This example wires those three loaders into the realistic generation scenario,
mirroring the host-Python scheduler loop of `sdxl-pipeline.py`: the text encoder
and VAE run on the host (CPU) while the heavy DiT denoiser is compiled with the
`tt` backend and runs on the Tenstorrent device for every denoise step. The
denoiser is a rectified-flow (flow-matching) model, so the sampler is a plain
Euler integration of the predicted velocity from noise (t=1) to data (t=0), with
classifier-free guidance over a batch of 2 (conditioned + null).

Geometry is kept at the native t2v configuration (480x832, 93 frames -> latent
[16, 24, 60, 104], DiT sequence length 37440) — the example never downscales the
resolution. Only the number of denoise steps is reduced (a faithful knob the
upstream demo also exposes) so the demonstration fits a CI budget.
"""

import argparse
import time
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.longcat_video.text_encoder.pytorch import (
    ModelLoader as TextEncoderLoader,
)
from third_party.tt_forge_models.longcat_video.transformer.pytorch import (
    ModelLoader as DenoiserLoader,
)
from third_party.tt_forge_models.longcat_video.vae.pytorch import (
    ModelLoader as VaeLoader,
)


class LongCatVideoPipeline:
    """Text-to-video pipeline driving the three LongCat-Video loaders."""

    def __init__(self):
        self.device = torch_xla.device()
        self.text_encoder_loader = TextEncoderLoader()
        self.denoiser_loader = DenoiserLoader()
        self.vae_loader = VaeLoader()

    def setup(self):
        """Load the three components; only the DiT denoiser goes to the device."""
        # Text encoder and VAE stay on the host (CPU), as in sdxl-pipeline.py.
        self.text_encoder = self.text_encoder_loader.load_model().eval()
        self.vae = self.vae_loader.load_model().eval()

        # The 13.6B DiT denoiser is the heavy, repeated compute -> compile for TT.
        self.denoiser = self.denoiser_loader.load_model().eval()
        self.denoiser = self.denoiser.to(device=self.device)
        self.denoiser.compile(backend="tt")

    def encode_prompt(self):
        """Embed the loader's default prompt for cross-attention conditioning."""
        enc_inputs = self.text_encoder_loader.load_inputs()
        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=enc_inputs["input_ids"],
                attention_mask=enc_inputs["attention_mask"],
            ).last_hidden_state  # [1, 512, 4096]
        # DiT expects [B, 1, N_token, C]; the null branch is a zero embedding
        # (text_tokens_zero_pad handles it), giving the batch-2 CFG layout.
        cond = text_embeds.unsqueeze(1)
        uncond = torch.zeros_like(cond)
        return torch.cat([uncond, cond], dim=0)  # [2, 1, 512, 4096]

    def generate(
        self,
        num_inference_steps: int = 20,
        cfg_scale: float = 7.5,
        seed: int = 0,
    ) -> torch.Tensor:
        """Run the flow-matching denoise loop on device and decode to RGB frames."""
        encoder_hidden_states = self.encode_prompt()

        # Native latent geometry: [16, 24, 60, 104] (16 ch, 24 frames, 60x104).
        nt, nh, nw = 24, 60, 104
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            1, 16, nt, nh, nw, generator=generator, dtype=torch.bfloat16
        )

        # Rectified-flow Euler schedule: sigma 1 (pure noise) -> 0 (data).
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)

        to_dev = lambda x: x.to(dtype=torch.bfloat16, device=self.device)
        enc_dev = to_dev(encoder_hidden_states)

        start = time.time()
        with torch.no_grad():
            for i in range(num_inference_steps):
                print(f"Denoise step {i + 1} of {num_inference_steps}")
                # batch-2 CFG: duplicate the latent for the cond/null branches.
                model_input = to_dev(latents.repeat(2, 1, 1, 1, 1))
                timestep = to_dev(torch.full((2,), float(sigmas[i] * 1000.0)))

                velocity = self.denoiser(
                    model_input,
                    timestep,
                    enc_dev,
                ).to("cpu", dtype=torch.float32)

                uncond_v, cond_v = velocity.chunk(2)
                v = uncond_v + cfg_scale * (cond_v - uncond_v)

                # Euler step along the flow (dt = sigma[i+1] - sigma[i] < 0).
                dt = (sigmas[i + 1] - sigmas[i]).to(torch.float32)
                latents = latents.to(torch.float32) + dt * v
                latents = latents.to(torch.bfloat16)
        print(f"Denoise loop time: {time.time() - start:.1f}s")

        # Decode the denoised latent to RGB video frames on the host.
        with torch.no_grad():
            video = self.vae(latents.to(torch.bfloat16))  # [1, 3, frames, H, W]
        return video.to(torch.float32)


def post_process_output(video: torch.Tensor, out_dir: str = "longcat_video_output"):
    """Save the decoded frames as PNGs and print the human-readable result path."""
    from PIL import Image

    standardize = lambda x: (torch.clamp(x / 2 + 0.5, 0.0, 1.0) * 255.0).to(torch.uint8)
    frames = standardize(video[0]).permute(1, 2, 3, 0).cpu().numpy()  # [F, H, W, C]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Save a handful of evenly-spaced frames so the artifact stays small.
    n = frames.shape[0]
    idxs = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1]))
    saved = []
    for fi in idxs:
        path = out / f"frame_{fi:03d}.png"
        Image.fromarray(frames[fi]).save(path)
        saved.append(str(path))

    print(f"Generated video: {n} frames at {frames.shape[2]}x{frames.shape[1]}")
    print(f"Saved sample frames: {saved}")
    return saved


def run_longcat_video(num_inference_steps: int = 20):
    """Build the pipeline, generate a video, and return the decoded tensor."""
    # Conservative compiler options; the full denoiser graph is very large.
    torch_xla.set_custom_compile_options({"optimization_level": 0})

    pipeline = LongCatVideoPipeline()
    pipeline.setup()
    return pipeline.generate(num_inference_steps=num_inference_steps)


def test_longcat_video():
    """Smoke test: the pipeline produces a finite video tensor of the right shape."""
    xr.set_device_type("TT")

    # A single denoise step is enough to exercise the full pipeline path.
    video = run_longcat_video(num_inference_steps=1)

    # Native t2v decode: 93 frames at 480x832, 3 channels.
    assert video.shape[0] == 1, f"Expected batch 1, got {video.shape[0]}"
    assert video.shape[1] == 3, f"Expected 3 channels, got {video.shape[1]}"
    assert torch.isfinite(video).all(), "Decoded video contains non-finite values"
    print(f"test_longcat_video OK: video shape {tuple(video.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="longcat_video_output")
    args = parser.parse_args()

    xr.set_device_type("TT")

    start = time.time()
    video = run_longcat_video(num_inference_steps=args.num_inference_steps)
    print(f"End-to-end generation time: {time.time() - start:.1f}s")
    post_process_output(video, out_dir=args.output_dir)
