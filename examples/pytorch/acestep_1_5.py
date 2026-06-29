# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ACE-Step 1.5 DiT denoiser example (text-to-music diffusion transformer).

ACE-Step 1.5 generates music latents with a flow-matching diffusion loop. The
per-step compute is the DiT backbone (``AceStepDiTModel``): given the current
noisy acoustic latent, the conditioning embeddings (text/lyric/timbre) and the
source-latent context, it predicts the flow-matching velocity used to advance
the latent one step toward clean audio.

This example compiles that DiT backbone with the ``tt`` backend, runs a single
denoising step on a TT device, and applies one Euler flow-matching update so the
result is a real denoised latent rather than a bare tensor. The full pipeline
would repeat this step over the sampler schedule before decoding the latent to
audio with the (separate) VAE.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.acestep.denoiser.pytorch import (
    ModelLoader,
    ModelVariant,
)


class DiTVelocity(torch.nn.Module):
    """Thin wrapper exposing the DiT's velocity prediction as a single tensor.

    The raw DiT ``forward`` returns ``(velocity, past_key_values)``; for a
    single denoising step the cache is unused, so we return only the velocity.
    """

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(
        self,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
    ):
        return self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False,
        )[0]


def acestep_denoise_step():
    """Run one ACE-Step 1.5 flow-matching denoising step on a TT device."""
    device = torch_xla.device()

    # Match the bringup's compiler configuration so the first compile is fast.
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    # Build the DiT backbone and a representative denoising-step input via the
    # tt_forge_models loader (10 s @ 25 Hz = 250 latent frames).
    loader = ModelLoader(ModelVariant.TURBO)
    model = DiTVelocity(loader.load_model()).eval()
    inputs = loader.load_inputs(batch_size=1)

    # Move model and tensor inputs to the device (drop the non-tensor flag).
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    model.compile(backend="tt")

    with torch.no_grad():
        velocity = model(
            hidden_states=inputs["hidden_states"],
            timestep=inputs["timestep"],
            timestep_r=inputs["timestep_r"],
            attention_mask=inputs["attention_mask"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_attention_mask=inputs["encoder_attention_mask"],
            context_latents=inputs["context_latents"],
        )

    # One Euler flow-matching update: integrate dx/dt = v from t toward t_r.
    dt = (inputs["timestep"] - inputs["timestep_r"]).view(-1, 1, 1)
    denoised_latent = inputs["hidden_states"] - dt * velocity

    return velocity.cpu(), denoised_latent.cpu()


def post_process_output(velocity, denoised_latent):
    """Print a human-readable summary of the denoising step."""
    print(f"DiT velocity prediction: shape {tuple(velocity.shape)}")
    v = velocity.float()
    print(f"  mean={v.mean():.4f}  std={v.std():.4f}  |max|={v.abs().max():.4f}")

    print(f"Denoised latent (one Euler step): shape {tuple(denoised_latent.shape)}")
    d = denoised_latent.float()
    print(f"  mean={d.mean():.4f}  std={d.std():.4f}  |max|={d.abs().max():.4f}")
    print(
        "This latent is one flow-matching step closer to clean audio; the full "
        "pipeline repeats the step over the schedule, then VAE-decodes to waveform."
    )


def test_acestep_1_5():
    """Test ACE-Step 1.5 DiT denoising step produces a finite, correctly-shaped latent."""
    xr.set_device_type("TT")

    velocity, denoised_latent = acestep_denoise_step()

    # The DiT predicts a velocity per acoustic latent channel (64) over the 250
    # latent frames; the Euler update keeps that shape.
    assert tuple(velocity.shape) == (1, 250, 64), f"unexpected shape {velocity.shape}"
    assert tuple(denoised_latent.shape) == (1, 250, 64)
    assert torch.isfinite(velocity).all(), "velocity has non-finite values"
    assert torch.isfinite(
        denoised_latent
    ).all(), "denoised latent has non-finite values"

    print("ACE-Step 1.5 denoising step produced a finite, correctly-shaped latent.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    velocity, denoised_latent = acestep_denoise_step()
    post_process_output(velocity, denoised_latent)
