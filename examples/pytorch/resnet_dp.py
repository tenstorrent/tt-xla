# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ResNet Data Parallel inference example.

This example demonstrates data-parallel inference where:
- The model is REPLICATED on each device
- The inputs are SHARDED along the batch dimension
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.resnet.image_classification.pytorch import ModelLoader, ModelVariant


def enable_spmd():
    """Enable torch_xla SPMD mode."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def resnet_dp():
    """Run ResNet with data parallel inference on TT devices."""
    num_devices = xr.global_runtime_device_count()

    # Load model using tt_forge_models loader
    loader = ModelLoader(ModelVariant.RESNET_50)
    model = loader.load_model()
    model = model.eval()

    # Create batch of inputs (batch_size must be divisible by num_devices)
    batch_size = num_devices * 4  # 4 samples per device
    inputs = loader.load_inputs(batch_size=batch_size)

    # Create 1D data-parallel mesh
    device_ids = np.arange(num_devices)
    mesh = Mesh(device_ids=device_ids, mesh_shape=(num_devices,), axis_names=("data",))

    device = torch_xla.device()

    # Move model to device (replicated)
    model = model.to(device=device)

    # Move inputs to device and mark for sharding along batch dim
    inputs = inputs.to(device=device)
    xs.mark_sharding(inputs, mesh, ("data", None, None, None))

    # Run inference
    with torch.no_grad():
        output = model(inputs)

    return output


def post_process_output(output):
    """Post-process and print top predictions."""
    logits = output.cpu()

    probabilities = torch.softmax(logits, dim=-1)
    top_5_probs, top_5_indices = torch.topk(probabilities, k=5)

    print(f"Processing {logits.shape[0]} batch items:")
    for batch_idx in range(min(logits.shape[0], 4)):  # Print first 4
        print(f"\nInput {batch_idx + 1} - Top 5 predictions:")
        for i in range(5):
            idx = top_5_indices[batch_idx, i].item()
            prob = top_5_probs[batch_idx, i].item() * 100
            print(f"{i+1}. class_{idx}: {prob:.2f}%")


def test_resnet_dp():
    """Test ResNet DP predictions have expected top-5 class ordering."""
    xr.set_device_type("TT")
    enable_spmd()

    output = resnet_dp()

    # Expected top-5 class indices (default COCO image produces these)
    expected_top5 = [281, 282, 285, 761, 721]

    logits = output.cpu()
    _, top_5_indices = torch.topk(logits, k=5, dim=-1)

    # Check all batch items have the expected top-5 ordering
    for batch_idx in range(logits.shape[0]):
        actual_top5 = top_5_indices[batch_idx].tolist()
        assert (
            actual_top5 == expected_top5
        ), f"Input {batch_idx}: expected top-5 {expected_top5}, got {actual_top5}"

    print("All batch items have expected top-5 class ordering.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")
    enable_spmd()

    output = resnet_dp()
    post_process_output(output)
