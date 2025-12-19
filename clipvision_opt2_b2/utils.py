# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
from . import ttnn_supplemental
import math
import torch


# Monkey-patch ttnn with ttnn_supplemental objects
ttnn.MeshShardDirection = ttnn_supplemental.MeshShardDirection
ttnn.MeshShardType = ttnn_supplemental.MeshShardType
ttnn.mesh_shard = ttnn_supplemental.mesh_shard
ttnn.collective_permute = ttnn_supplemental.collective_permute


# Wrapper to abstract const-eval logic out of runtime funcs to keep them
# cleaner. Invokes constEvalFunc iff outputs is empty.
def constEvalFuncWrapper(constEvalFunc, inputs, outputs):
    if not outputs:
        outputs = constEvalFunc(inputs)
    return outputs


# Wrapper to abstract const-eval logic out of runtime funcs to keep them
# cleaner. Invokes constEvalFunc iff outputs is empty.
# This is an overload of constEvalFuncWrapper for const-eval functions that
# take zero arguments.
def constEvalFuncWrapperZeroArg(constEvalFunc, outputs):
    if not outputs:
        outputs = constEvalFunc()
    return outputs


def get_scalar_from_tensor(tensor: ttnn.Tensor) -> int:
    assert tensor.logical_volume() == 1, "expected scalar tensor"
    assert tensor.dtype == ttnn.DataType.UINT32, "expected uint32 tensor"

    host_tensor = ttnn.from_device(tensor)
    return host_tensor.item()


def load_tensor(file_path: str, layout, dtype, device, memory_config) -> ttnn.Tensor:
    loaded_tensor = ttnn.load_tensor(file_path)

    assert loaded_tensor.device() is None, "loaded tensor must be on host"

    if loaded_tensor.layout != layout:
        loaded_tensor = ttnn.to_layout(loaded_tensor, layout)
    if loaded_tensor.dtype != dtype:
        loaded_tensor = ttnn.to_dtype(loaded_tensor, dtype)
    if device is not None:
        loaded_tensor = ttnn.to_device(loaded_tensor, device, memory_config)

    return loaded_tensor


def calculate_pcc(x, y):
    # This function calculates the PCC between two torch tensors

    # Assert both are torch tensors
    assert isinstance(x, torch.Tensor), "x must be a torch tensor"
    assert isinstance(y, torch.Tensor), "y must be a torch tensor"

    if x.shape != y.shape:
        raise ValueError(f"Shapes of x and y must be the same, but got {x.shape} and {y.shape}")

    # Calculate PCC
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom
