# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pi-0.5 (pi05_base) vision-language-action robot control example.

Pi-0.5 is a generalist robot policy: given a few camera views, the robot's
proprioceptive state and a natural-language task instruction, it predicts the
next chunk of low-level actions to execute. This example drives a single
real-world control step end-to-end on a TT device:

    observation (3 camera frames + state + "pick up the cube ...")
        -> tokenized prompt + preprocessed images  (loader)
        -> flow-matching action sampler on device   (torch.compile backend="tt")
        -> the next action vector to send to the robot

Weights and the synthetic observation come straight from the tt-forge-models
``pi_0_5`` loader, so the example needs no external dataset. The model runs in
bfloat16 on device (mirroring the tt-xla runner), and a fixed starting noise
tensor from the loader keeps the sampled action deterministic.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.pi_0_5.pytorch import ModelLoader, ModelVariant


# --------------------------------
# Pi-0.5 single control-step inference
# --------------------------------
def pi05_base():
    """Predict the next robot action on a TT device via the loader API."""
    loader = ModelLoader(ModelVariant.BASE)

    # Weights in bfloat16 and float inputs (images/noise) cast to match, as the
    # tt-xla runner does. Integer language tokens and boolean masks are left as-is.
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    images, img_masks, lang_tokens, lang_masks, noise = loader.load_inputs(
        dtype_override=torch.bfloat16
    )

    device = torch_xla.device()

    # Move the model and every input to the device. Images and camera masks are
    # per-camera lists; the language tokens/masks and noise are single tensors.
    model = model.to(device)
    images = [img.to(device) for img in images]
    img_masks = [m.to(device) for m in img_masks]
    lang_tokens = lang_tokens.to(device)
    lang_masks = lang_masks.to(device)
    noise = noise.to(device)

    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        action = compiled_model(images, img_masks, lang_tokens, lang_masks, noise=noise)

    # Un-normalize the sampled action back into the robot's action space.
    return loader.postprocess(action.cpu())


def post_process_output(action):
    """Print the predicted next action vector in a human-readable form."""
    action = action.float().cpu().squeeze(0)

    print("=" * 80)
    print("Task: pick up the cube and place it on the plate")
    print("Cameras: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb (3x224x224)")
    print("-" * 80)
    print(f"Predicted next action ({action.shape[0]}-dim):")
    for i, v in enumerate(action.tolist()):
        print(f"  a[{i:2d}] = {v:+.4f}")
    print("=" * 80)


def test_pi05_base():
    """Pi-0.5 predicts a finite action vector of the expected shape on device."""
    xr.set_device_type("TT")

    action = pi05_base()

    # One control step -> one action of the model's full action dimension.
    assert action.shape == (1, 32), f"unexpected action shape {tuple(action.shape)}"
    assert torch.isfinite(action).all(), "predicted action has non-finite values"

    print("Pi-0.5 produced a finite 32-dim action for the control step.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    action = pi05_base()
    post_process_output(action)
