# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel Qwen-Image MMDiT denoiser on a multi-chip Tenstorrent device.

Qwen-Image is a text-to-image diffusion pipeline (Qwen2.5-VL text encoder, an
``AutoencoderKLQwenImage`` VAE, and the ``QwenImageTransformer2DModel`` MMDiT
denoiser driven by a FlowMatchEulerDiscreteScheduler). The denoiser is the heavy
per-step compute — ~20B params (~40GB in bf16) that does not fit a single 32GB
Blackhole chip — so the realistic device scenario is one MMDiT denoising step
sharded tensor-parallel across the chips.

This example builds the denoiser and its native-resolution (1024x1024) inputs via
the tt-forge-models loader, lays it out with the loader's Megatron column->row
shard spec (``get_mesh_config`` / ``load_shard_spec``), compiles it with the "tt"
backend, and runs a single denoising step, mirroring ``qwen3_tp.py``. The prompt
embeddings are the loader's synthetic stand-in for the Qwen2.5-VL conditioning
(the text encoder is a separate component loader), so this validates the denoiser
compute and sharding rather than producing a final image.
"""
import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.qwen_image.transformer.pytorch import (
    ModelLoader,
    ModelVariant,
)


def _to_device(inputs, device):
    """Move the tensor entries of the loader input dict to the device.

    ``load_inputs`` mixes tensors (latents, prompt embeds, timestep) with plain
    Python values (``img_shapes``, ``guidance``, ``return_dict``) that the
    denoiser forward consumes as-is, so only the tensors are relocated.
    """
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }


def qwen_image_tp():
    # Shardy is the sharding path the loader's column/row spec is written for.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Build the denoiser and native-resolution inputs through the loader API.
    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Mesh for tensor-parallel sharding (1 x num_devices: batch x model).
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)

    # Move model + inputs to device, then apply the loader's shard spec.
    device = torch_xla.device()
    model = model.to(device)
    inputs = _to_device(inputs, device)

    shard_specs = loader.load_shard_spec(model)
    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # Compile and run one MMDiT denoising step.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        # return_dict=False (set by the loader) => forward returns a tuple whose
        # first entry is the predicted velocity over the packed image latents.
        output = compiled_model(**inputs)

    return output[0].cpu()


def post_process_output(velocity):
    """Print a human-readable summary of the denoiser's per-step output."""
    finite = torch.isfinite(velocity).all().item()
    print("Qwen-Image MMDiT denoising step (tensor-parallel):")
    print(f"  predicted velocity shape : {tuple(velocity.shape)}")
    print(f"  dtype                    : {velocity.dtype}")
    print(f"  all finite               : {finite}")
    print(f"  mean / std               : {velocity.float().mean():.4f} / "
          f"{velocity.float().std():.4f}")
    print(f"  L2 norm                  : {velocity.float().norm():.2f}")


def test_qwen_image():
    """One tensor-parallel MMDiT denoising step produces a sane velocity tensor.

    The denoiser predicts a velocity over the packed image latents, so the output
    shape must match the packed-latent input ([B, img_seq_len, in_channels]) and
    every element must be finite.
    """
    xr.set_device_type("TT")

    velocity = qwen_image_tp()

    # Packed-latent sequence length from the loader's public native resolution
    # (VAE downsamples by 8, the pipeline packs 2x2 patches on top).
    latent_h = ModelLoader.HEIGHT // 8 // 2
    latent_w = ModelLoader.WIDTH // 8 // 2
    expected_seq_len = latent_h * latent_w

    assert velocity.ndim == 3, f"Expected [B, seq, C] output, got {tuple(velocity.shape)}"
    assert velocity.shape[0] == 1, f"Expected batch 1, got {velocity.shape[0]}"
    assert velocity.shape[1] == expected_seq_len, (
        f"Unexpected packed-latent length {velocity.shape[1]}, "
        f"expected {expected_seq_len}"
    )
    assert torch.isfinite(velocity).all(), "Denoiser output contains non-finite values"

    print("Qwen-Image tensor-parallel denoising step validated.")


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    velocity = qwen_image_tp()
    post_process_output(velocity)
