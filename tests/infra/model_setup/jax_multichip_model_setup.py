# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""JAX multichip model setup helpers extracted from DynamicJaxMultiChipModelTester."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import jax
from flax import linen, nnx
from infra.connectors import DeviceConnectorFactory, JaxDeviceConnector
from infra.runners import JaxDeviceRunner
from infra.utilities import Framework, PyTree, ShardingMode
from infra.workloads import JaxMultichipWorkload, Workload
from jax.sharding import PartitionSpec


def initialize_meshes(
    device_runner: JaxDeviceRunner, mesh_shape: tuple, axis_names: tuple
) -> Tuple[jax.sharding.Mesh, jax.sharding.Mesh]:
    """Returns (device_mesh, cpu_mesh) for the given mesh configuration."""
    assert isinstance(device_runner, JaxDeviceRunner) and isinstance(
        device_runner.connector, JaxDeviceConnector
    )
    device_mesh = device_runner.connector.get_tt_device_mesh(mesh_shape, axis_names)
    cpu_mesh = device_runner.connector.get_cpu_device_mesh(mesh_shape, axis_names)
    return device_mesh, cpu_mesh


def create_jax_multichip_workload(
    workload: Workload,
    device_mesh: jax.sharding.Mesh,
    in_specs: tuple,
    out_spec: PartitionSpec = None,
    sharding_mode: ShardingMode = ShardingMode.INPUTS_AND_MODULE,
    model=None,
) -> JaxMultichipWorkload:
    """Creates a JaxMultichipWorkload from a single-chip Workload and mesh config."""
    assert workload.is_jax, "Workload must be JAX workload"

    if out_spec is None:
        out_spec = PartitionSpec()  # Replicated by default

    return JaxMultichipWorkload(
        executable=workload.executable,
        compiled_executable=workload.compiled_executable,
        model=model,
        args=workload.args,
        kwargs=workload.kwargs,
        static_argnames=workload.static_argnames,
        device_mesh=device_mesh,
        in_specs=in_specs,
        out_spec=out_spec,
        sharding_mode=sharding_mode,
    )


def get_forward_method_arg_specs(
    model,
    input_parameters_partition_specs: PyTree,
    input_activations_partition_specs,
) -> tuple:
    """Returns partition specs for the forward method arguments."""
    if isinstance(model, (linen.Module, nnx.Module)):
        return (
            input_parameters_partition_specs,
            *input_activations_partition_specs,
        )
    return ()


def get_number_of_tt_devices() -> int:
    """Returns the number of available TT devices."""
    connector = DeviceConnectorFactory.create_connector(Framework.JAX)
    return connector.get_number_of_tt_devices()
