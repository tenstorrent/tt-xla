# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pi-0.5 (Pi05) vision-language-action inference example.

Pi-0.5 is a generalist robot policy: given a few camera views and a natural
language instruction (the proprioceptive robot state is discretized into that
instruction by the pre-processor), it predicts a chunk of low-level robot
actions via a flow-matching denoising loop.

This example drives the policy end-to-end on a single TT device through the
tt-forge-models loader API:
  - the loader builds the PI05Policy and a real LIBERO observation
    (two camera streams + the tokenized instruction + a deterministic noise
    tensor for the flow-matching sampler),
  - the policy is compiled with the "tt" backend and run on device,
  - the predicted action vector (one step of the action chunk) is printed.

Single-chip scenario (n150/p150): a plain compile + run, modelled after
`compiler_options.py` / `resnet_dp.py` but for a multi-input action policy.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.pi_05.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)


def _to_device(x, device):
    """Move a tensor (or a list of tensors, as the loader returns for the
    per-camera image / image-mask inputs) onto the TT device."""
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(e, device) for e in x)
    return x.to(device)


# --------------------------------
# Pi-0.5 action-prediction scenario
# --------------------------------
def pi05():
    """Run Pi-0.5 action prediction on a TT device and return the action chunk's
    first predicted action (shape [batch, action_dim])."""
    device = torch_xla.device()

    # Build the policy and a real LIBERO observation via the loader's public API.
    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model().eval()
    images, img_masks, lang_tokens, lang_masks, noise = loader.load_inputs()

    # Move the model and every input (including the per-camera lists) to device.
    model = model.to(device)
    images = _to_device(images, device)
    img_masks = _to_device(img_masks, device)
    lang_tokens = lang_tokens.to(device)
    lang_masks = lang_masks.to(device)
    noise = noise.to(device)

    # Compile for the TT backend and sample an action chunk on device.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        action = compiled_model(images, img_masks, lang_tokens, lang_masks, noise=noise)

    return action.cpu()


def post_process_output(action):
    """Print the predicted robot action in a human-readable form."""
    action = action.float()
    print(f"Predicted action vector (shape {tuple(action.shape)}):")
    values = action.flatten().tolist()
    preview = ", ".join(f"{v:+.4f}" for v in values[:8])
    print(f"  first action of the chunk: [{preview}{', ...' if len(values) > 8 else ''}]")
    print(f"  L2 norm: {action.flatten().norm().item():.4f}")


def test_pi05():
    """Smoke test: Pi-0.5 produces a finite action of the expected shape on device."""
    xr.set_device_type("TT")

    action = pi05()

    # One step of the action chunk: [batch_size, action_dim].
    assert action.ndim == 2, f"expected a 2D action, got shape {tuple(action.shape)}"
    assert action.shape[-1] == 32, f"expected action_dim 32, got {action.shape[-1]}"
    assert torch.isfinite(action).all(), "action contains non-finite values"

    print(f"Pi-0.5 produced a finite action of shape {tuple(action.shape)}.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    action = pi05()
    post_process_output(action)
