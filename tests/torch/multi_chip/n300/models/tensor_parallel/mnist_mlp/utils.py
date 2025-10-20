# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from tests.torch.multi_chip.utils import replicate_bias, shard_bias


def apply_tensor_parallel_sharding_mnist_linear(
    model: nn.Module, mesh, *, move_to_device: bool = True, strict_bias: bool = False
):
    """
    Shard all Linear weights in alternating fashion:
        - fc1: shard on outer dimension (output features)
        - fc2: shard on inner dimension (input features)
        - fc3: shard on outer dimension (output features)

    Biases are sharded or replicated depending on sharding of respective linear weight.
    Bias b: shape (out_features,), so:
    When linear weight is sharded along outer dimension, bias is also sharded (along outer dimension).
    When linear weight is sharded along inner dimension, bias is replicated.

    Note:
    PyTorch's nn.Linear(in_features, out_features) stores the weight as (out_features, in_features).
    Therefore, sharding matmul along outer dimension means sharding rows, and sharding along inner dimension means sharding columns.

    Sharding spec is tuple that maps each tensor dim to mesh axis or None.
    sharding spec for outer sharding: ("model", None)
    sharding spec for inner sharding: (None, "model")
    """
    if move_to_device:
        model.to(torch_xla.device())

    # fc1: weight [hidden, input] -> shard rows (output features)
    xs.mark_sharding(model.fc1.weight, mesh, ("model", None))
    shard_bias(model.fc1.bias, mesh)

    # fc2: weight [hidden, hidden] -> shard cols (input features)
    xs.mark_sharding(model.fc2.weight, mesh, (None, "model"))
    replicate_bias(model.fc2.bias, mesh)

    # # fc3: weight [num_classes, hidden] -> shard rows (out_features)
    xs.mark_sharding(model.fc3.weight, mesh, ("model", None))
    shard_bias(model.fc3.bias, mesh)


def tensor_parallel_inference_mnist(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    mesh=None,
):
    """
    Run MNISTLinear in tensor-parallel mode.
    Inputs are replicated to all model shards.
    """
    if mesh is None:
        num_devices = xr.global_runtime_device_count()
        device_ids = np.arange(num_devices)
        mesh = Mesh(
            device_ids=device_ids,
            mesh_shape=(1, num_devices),
            axis_names=("batch", "model"),
        )

    device = torch_xla.device()
    model = model.to(device=device, dtype=dtype).eval()

    # Apply sharding to parameters
    # Use apply_tensor_parallel_sharding_mnist_linear for manual marking
    apply_tensor_parallel_sharding_mnist_linear(model, mesh, move_to_device=False)

    # Replicate inputs to all devices (no sharding along 'model')
    inputs = inputs.to(device=device, dtype=dtype)
    # 2D input (N, D) -> (None, None) means no sharding (replicated) on both dims
    xs.mark_sharding(inputs, mesh, (None, None))

    with torch.no_grad():
        outputs = model(inputs)

    return outputs
