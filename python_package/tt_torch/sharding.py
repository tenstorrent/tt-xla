# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
User-facing sharding APIs for applying sharding constraints to intermediate tensors.

Example:
    >>> from tt_torch import sharding_constraint_hook
    >>> mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    >>> sharding_constraint_hook(model.embed_tokens, mesh, ("batch", None, None))
"""

import torch

# Generic mesh axis index placeholder prefix.
# tt-mlir will replace these with actual axis names from the mesh definition.
# e.g., "mesh_idx_0" -> "x" (or "_axis_0" depending on mesh setup)
_MESH_IDX_PREFIX = "mesh_idx_"


def _partition_spec_to_sdy_sharding(mesh, partition_spec) -> str:
    """
    Convert a partition_spec to an sdy.sharding string.

    Uses generic placeholders (mesh_idx_0, mesh_idx_1) that tt-mlir will
    replace with actual axis names from the mesh definition.

    Example:
        partition_spec = ("batch", None, None)
        mesh.axis_names = ("batch", "model")
        → '#sdy.sharding_per_value<[<@mesh, [{"mesh_idx_0"}, {}, {}]>]>'
    """
    dim_shardings = []
    for axis in partition_spec:
        if axis is None:
            dim_shardings.append("{}")
        elif isinstance(axis, str):
            # Map axis name to mesh index placeholder (e.g., "batch" -> "mesh_idx_0")
            try:
                axis_idx = mesh.axis_names.index(axis)
                dim_shardings.append(f'{{"{_MESH_IDX_PREFIX}{axis_idx}"}}')
            except ValueError:
                dim_shardings.append("{}")
        elif isinstance(axis, int):
            dim_shardings.append(f'{{"{_MESH_IDX_PREFIX}{axis}"}}')
        else:
            dim_shardings.append("{}")

    dims_str = ", ".join(dim_shardings)
    return f"#sdy.sharding_per_value<[<@mesh, [{dims_str}]>]>"


def sharding_constraint_hook(module, mesh, partition_spec):
    """
    Apply a sharding constraint to a module's output.

    This is the recommended way to apply sharding constraints to intermediate tensors
    in a torch.compile-compatible manner.

    Args:
        module: The nn.Module whose output should be sharded
        mesh: The mesh object describing device topology
        partition_spec: A tuple specifying how each dimension should be sharded.
            Use axis names (e.g., "batch", "model") or None for replicated dimensions.

    Returns:
        torch.utils.hooks.RemovableHandle: A handle that can be used to remove the hook
            by calling handle.remove()

    Raises:
        TypeError: If module is not an nn.Module
        ValueError: If mesh or partition_spec is invalid

    Example:
        >>> from tt_torch import sharding_constraint_hook
        >>> mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
        >>> handle = sharding_constraint_hook(model.embed_tokens, mesh, ("batch", None, None))
        >>> # Later, to remove the hook:
        >>> handle.remove()
    """
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(module).__name__}")

    if mesh is None:
        raise ValueError("mesh cannot be None")

    if partition_spec is None:
        raise ValueError("partition_spec cannot be None")

    if not hasattr(mesh, "axis_names"):
        raise ValueError("mesh must have 'axis_names' attribute")

    # Convert to sdy.sharding string at hook creation time
    sdy_sharding = _partition_spec_to_sdy_sharding(mesh, partition_spec)

    def hook(mod, input, output):
        return torch.ops.tt.sharding_constraint(output, sdy_sharding)

    return hook
