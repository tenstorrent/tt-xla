# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import operator
import warnings
from functools import partial
from typing import Any, Dict, List, Literal, Mapping, Sequence, Union

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch.fx import GraphModule, Node
from torch_xla.distributed.spmd import Mesh


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
