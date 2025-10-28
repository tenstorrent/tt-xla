# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


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


def _normalize_dim(dim: int, rank: int) -> int:
    # Convert negative dim to positive
    if dim < 0:
        dim += rank
    if not (0 <= dim < rank):
        raise ValueError(f"batch_dim {dim} out of range for rank={rank}")
    return dim


def _make_dp_sharding_spec(rank: int, batch_dim: int):
    "Create data-parallel sharding spec for given rank and batch_dim."
    # Tuple like (None, None, ..., "data", ..., None) with length = rank
    spec = [None] * rank
    spec[batch_dim] = "data"
    return tuple(spec)


def _map_structure(fn, obj):
    # Minimal tree-map supporting Tensor / list / tuple / dict
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map_structure(fn, x) for x in obj)
    if isinstance(obj, list):
        return [_map_structure(fn, x) for x in obj]
    if isinstance(obj, dict):
        return {k: _map_structure(fn, v) for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type {type(obj)} in inputs.")


def mark_dp_input_sharding(
    input: torch.Tensor, batch_dim, allow_uneven_shards, num_devices, mesh
) -> torch.Tensor:
    """
    Mark a tensor for data-parallel sharding along the specified batch dimension.

    Validates that the batch dimension size is compatible with the number of devices
    and applies the appropriate SPMD sharding annotation for data parallelism.

    Args:
        input: Input tensor (should already be on XLA device)
        batch_dim: Dimension to shard across devices (supports negative indexing)
        allow_uneven_shards: If True, allow batch size not divisible by num_devices
        num_devices: Number of devices in the mesh
        mesh: XLA SPMD mesh for sharding

    Returns:
        The input tensor with sharding annotations applied
    """
    rank = input.dim()
    bd = _normalize_dim(batch_dim, rank)

    # Divisibility check
    if not allow_uneven_shards and (input.size(bd) % num_devices != 0):
        raise ValueError(
            f"Batch size {input.size(bd)} along dim={bd} not divisible by #devices={num_devices}. "
            f"Set allow_uneven_shards=True if you know what you're doing."
        )
    if allow_uneven_shards and (input.size(bd) % num_devices != 0):
        warnings.warn(
            f"[allow_uneven_shards] Batch size {input.size(bd)} not divisible by {num_devices}. "
            "Depending on your XLA build, this may cause padding or uneven partition behavior."
        )

    # Sharding spec: ("data", None, None, ...)
    spec = _make_dp_sharding_spec(rank, bd)
    xs.mark_sharding(input, mesh, spec)
    return input


def data_parallel_inference_generic(
    model: nn.Module,
    inputs: Any,
    *,
    batch_dim: int = 0,
    allow_uneven_shards: bool = False,
    mesh: "Mesh | None" = None,
) -> Any:
    """
    Run data-parallel inference on XLA for arbitrary input structures.

    - model is REPLICATED on each device
    - inputs are SHARDED along `batch_dim` (per-tensor)
    - supports nested inputs: Tensor / tuple / list / dict
    - returns outputs gathered to one device.

    Args:
        batch_dim: which dim is the batch for *each* input Tensor (negatives allowed).
        allow_uneven_shards: if False, enforce batch size divisible by num devices.
        mesh: optional Mesh. If None, a 1D data mesh of size world_size is created.

    Returns:
        Model outputs.
    """

    # Mesh/device setup
    num_devices = xr.global_runtime_device_count()
    if num_devices <= 0:
        raise RuntimeError("No XLA devices found.")

    if mesh is None:
        device_ids = np.arange(num_devices)
        mesh = Mesh(
            device_ids=device_ids, mesh_shape=(num_devices,), axis_names=("data",)
        )

    device = torch_xla.device()

    # Move model to device
    model = model.to(device=device).eval()

    # Move inputs to device and mark for sharding
    def move_and_shard(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device=device)
        return mark_dp_input_sharding(
            t, batch_dim, allow_uneven_shards, num_devices, mesh
        )

    sharded_inputs = _map_structure(move_and_shard, inputs)

    # Inference (no grad)
    with torch.no_grad():
        outputs = model(sharded_inputs)

    return outputs


def replicate_bias(bias, mesh):
    """
    Replicate a bias tensor across all devices in the mesh.

    Use this when the corresponding weight is sharded along the inner dimension (columns),
    meaning the bias needs to be replicated on all devices since each device computes
    partial results that need the full bias vector.

    Args:
        bias: Bias tensor to replicate (can be None)
        mesh: XLA SPMD mesh defining the device topology
    """
    if bias is None:
        return
    xs.clear_sharding(bias)
    xs.mark_sharding(bias, mesh, (None,))  # replicate (no axis mapping)


def shard_bias(bias, mesh):
    """
    Shard a bias tensor along the 'model' axis.

    Use this when the corresponding weight is sharded along the outer dimension (rows),
    meaning each device computes a subset of the output features and needs only
    the corresponding slice of the bias vector. Raises ValueError if bias size
    is not evenly divisible by the number of devices.

    Args:
        bias: Bias tensor to shard (can be None)
        mesh: XLA SPMD mesh defining the device topology

    Raises:
        ValueError: If bias length is not divisible by the number of devices
    """
    num_devices = xr.global_runtime_device_count()
    xs.clear_sharding(bias)

    if bias is None:
        return
    if bias.numel() % num_devices == 0:
        # bias can be sharded
        xs.mark_sharding(bias, mesh, ("model",))
    else:
        msg = (
            f"Bias length {bias.numel()} not divisible by #devices={num_devices}; "
            "replicating bias instead of sharding."
        )
        raise ValueError(msg)
