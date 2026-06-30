# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 (Pi05) Vision-Language-Action single-forward example.

PI0.5 is a Vision-Language-Action (VLA) policy from Physical Intelligence,
distributed through the ``lerobot`` library. It pairs a PaliGemma backbone
(Gemma-2B language model + SigLIP vision tower) with a Gemma-300M "action
expert" and predicts continuous robot action chunks via flow matching.

This example drives the model's compute-dominant component — the vision tower
plus the joint PaliGemma/expert transformer — through one flow-matching step on
a Tenstorrent device, producing the predicted velocity field ``v_t`` of shape
``[batch, chunk_size, max_action_dim]``. The full ``sample_actions`` inference
path iterates this forward inside a host-Python denoising loop with
data-dependent control flow; a single step is the deterministic graph that
compiles and runs on device, so that is what this reference example exercises.

The model and a deterministic sample observation batch are built through the
tt-forge-models ``ModelLoader`` public API, mirroring ``resnet_dp.py``.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.pi05.action_prediction.pytorch import (
    ModelLoader,
    ModelVariant,
)


def _to_device(inputs, device):
    """Move every tensor in the loader's input dict onto the TT device."""
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def pi05():
    """Run one PI0.5 flow-matching forward step on a TT device."""
    # Build the model and a deterministic sample observation via the loader.
    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model().eval()
    inputs = loader.load_inputs(batch_size=1)

    # Connect the device, replicate the model and inputs onto it.
    device = torch_xla.device()
    model = model.to(device)
    inputs = _to_device(inputs, device)

    # Compile the single-step forward for the TT backend and run it.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        v_t = compiled_model(**inputs)

    return v_t


def post_process_output(v_t):
    """Print a human-readable summary of the predicted action velocity field."""
    v_t = v_t.cpu().float()
    batch, chunk_size, action_dim = v_t.shape

    print(
        f"PI0.5 flow-matching velocity field v_t: "
        f"batch={batch}, chunk_size={chunk_size}, action_dim={action_dim}"
    )
    print(f"Instruction: {ModelLoader.sample_instruction!r}")
    print(f"Global velocity L2 norm: {v_t.norm().item():.4f}")

    # Show the predicted velocity for the first action step of the chunk.
    first_step = v_t[0, 0]
    preview = ", ".join(f"{x:.4f}" for x in first_step[:8].tolist())
    print(f"Predicted velocity for action step 0 (first 8 dims): [{preview}, ...]")


def test_pi05():
    """Guard the PI0.5 example: output has the expected shape and is finite."""
    xr.set_device_type("TT")

    # Action-chunk geometry of the lerobot/pi05_base checkpoint:
    # chunk_size=50 future action steps, max_action_dim=32.
    expected_shape = (1, 50, 32)

    v_t = pi05().cpu().float()

    assert v_t.shape == expected_shape, (
        f"unexpected velocity field shape {tuple(v_t.shape)}, "
        f"expected {expected_shape}"
    )
    assert torch.isfinite(v_t).all(), "velocity field contains non-finite values"

    print(f"PI0.5 forward produced a finite {tuple(v_t.shape)} velocity field.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    output = pi05()
    post_process_output(output)
