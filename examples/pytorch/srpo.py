# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO (tencent/SRPO) tensor-parallel denoising-step example.

SRPO is a FLUX.1-dev fine-tune from Tencent Hunyuan that ships only the ~12B
``FluxTransformer2DModel`` denoiser. That transformer runs out of DRAM on a
single chip, so it is brought up across multiple Tenstorrent chips with
Megatron-style 1-D tensor parallelism over a ``(None, "model")`` mesh (the same
topology the tt-xla ``srpo/pytorch-Base-tensor_parallel-inference`` gate runs).

This example drives that topology end-to-end through the tt-forge-models loader:
it shards the SRPO transformer over every visible chip using the loader's own
mesh hooks (``get_mesh_config`` / ``load_shard_spec``) and runs one flow-matching
denoising step on device. One step is the unit the multi-chip bringup validated;
a full 1024x1024 multi-step generation is a much heavier cold compile that does
not fit the example stage's time budget (and must never be run at a reduced
resolution just to make it cheaper).
"""
import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.srpo.pytorch import ModelLoader, ModelVariant


def _enable_spmd():
    """Enable torch_xla SPMD with the shardy path the TT PJRT compiler expects."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def srpo_tp():
    """Shard the SRPO/FLUX transformer tensor-parallel and run one denoise step.

    Returns:
        torch.Tensor: the predicted flow-matching velocity for one denoising
        step, shape ``[batch, img_seq, in_channels]`` on CPU.
    """
    _enable_spmd()
    num_devices = xr.global_runtime_device_count()

    # Build the model and its inputs through the loader's public API.
    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Megatron-1D mesh, straight from the loader's hook: axis "model" is the TP
    # degree, the leading axis is unnamed (no data parallelism).
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, mesh_names)
    print(f"Sharding SRPO transformer over {num_devices} chips, mesh {mesh_shape}.")

    # Move weights and inputs to the TT device.
    device = torch_xla.device()
    model = model.to(device)
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    # Apply the loader's tensor-parallel shard spec (params absent from the map
    # are replicated across the mesh).
    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # Compile with the TT backend and run one denoising step.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    noise_pred = output[0] if isinstance(output, tuple) else output.sample
    return noise_pred.to("cpu")


def post_process_output(noise_pred: torch.Tensor):
    """Print a human-readable summary of the denoising-step prediction."""
    finite = torch.isfinite(noise_pred).all().item()
    flat = noise_pred.flatten().to(torch.float32)
    print("\nSRPO tensor-parallel denoising step")
    print(f"  prompt          : {ModelLoader.prompt!r}")
    print(f"  velocity shape  : {tuple(noise_pred.shape)}  dtype {noise_pred.dtype}")
    print(f"  all finite      : {finite}")
    print(
        "  value stats     : "
        f"min {flat.min():.4f}  max {flat.max():.4f}  "
        f"mean {flat.mean():.4f}  std {flat.std():.4f}  |v| {flat.norm():.2f}"
    )


def test_srpo():
    """Guard the example: the sharded denoising step must produce finite output
    with the transformer's expected ``[batch, img_seq, in_channels]`` shape."""
    xr.set_device_type("TT")

    noise_pred = srpo_tp()

    assert torch.isfinite(noise_pred).all(), "Denoising-step prediction is non-finite"
    assert noise_pred.dim() == 3, f"Expected a rank-3 prediction, got {noise_pred.dim()}"
    assert noise_pred.shape[0] == 1, f"Expected batch 1, got {noise_pred.shape[0]}"
    print(f"\nOK: finite prediction with shape {tuple(noise_pred.shape)}")


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    noise_pred = srpo_tp()
    post_process_output(noise_pred)
