# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Sequence

import jax
from flax import linen
from infra.connectors import DeviceConnector
from infra.utilities import Device, Tensor
from infra.workloads import JaxMultichipWorkload, Workload
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import PyTree

from .device_runner import DeviceRunner


class JaxDeviceRunner(DeviceRunner):
    """Device runner used with JAX."""

    @property
    def connector(self) -> DeviceConnector:
        """Exposed connector to easily reach its methods."""
        return self._device_connector

    # @override
    def _run_on_device(self, workload: Workload, device: Device) -> Tensor:

        with jax.default_device(device):
            return workload.execute()

    # @override
    def _safely_put_workload_on_device(
        self, workload: Workload, device: Device
    ) -> Workload:
        """
        Puts workload's args and kwargs on device only if `jax.device_put` supports it
        and returns new workload which is "on device".

        `jax.device_put` by docs accepts
        ``An array, scalar, or (nested) standard Python container thereof``
        which is too vague and not easy to check. In best case, has to be done
        recursively.

        To avoid that, we try to `jax.device_put` arg or kwarg, and if it doesn't
        succeed, we leave it as is.
        """
        assert workload.is_jax, "Workload must be JAX workload to put on device"

        args_on_device = []
        kwargs_on_device = {}

        fn_params = list(inspect.signature(workload.executable).parameters.keys())

        for i, arg in enumerate(workload.args):
            if fn_params[i] not in workload.static_argnames:
                try:
                    args_on_device.append(jax.device_put(arg, device))
                except:
                    args_on_device.append(arg)
            else:
                args_on_device.append(arg)

        for key, value in workload.kwargs.items():
            if key not in workload.static_argnames:
                try:
                    kwargs_on_device[key] = jax.device_put(value, device)
                except:
                    kwargs_on_device[key] = value
            else:
                kwargs_on_device[key] = value

        return Workload(
            framework=workload.framework,  # Unchanged.
            executable=workload.executable,  # Unchanged.
            compiled_executable=workload.compiled_executable,  # Unchanged.
            args=args_on_device,
            kwargs=kwargs_on_device,
            static_argnames=workload.static_argnames,
        )

    def run_on_multichip_device(
        self, multichip_workload: JaxMultichipWorkload
    ) -> Tensor:
        """
        Runs `multichip_workload` on a multichip device.
        Depending on the sharding mode, we might put the workload directly on device, or
        leave it to jax to infer on the fly.
        """

        if multichip_workload.sharding_mode.requires_device_put:
            sharded_workload = self._put_multichip_workload_on_device(
                multichip_workload
            )
            return sharded_workload.execute()
        else:
            return multichip_workload.execute()

    def _put_multichip_workload_on_device(
        self,
        multichip_workload: JaxMultichipWorkload,
    ) -> JaxMultichipWorkload:
        """Gives the workload inputs shardings, necessary for multichip workloads."""
        args_on_device = []
        spec_index = 0
        for arg in multichip_workload.args:
            device_arg = self._put_sharded_arg_on_multichip_device(
                arg,
                multichip_workload.device_mesh,
                multichip_workload.in_specs[spec_index],
            )
            # Increment the spec index if the argument was put on device.
            if device_arg is not arg:
                spec_index += 1
            args_on_device.append(device_arg)

        kwargs_on_device = {}
        for key, arg in multichip_workload.kwargs.items():
            device_arg = self._put_sharded_arg_on_multichip_device(
                arg,
                multichip_workload.device_mesh,
                multichip_workload.in_specs[spec_index],
            )
            # Increment the spec index if the argument was put on device.
            if device_arg is not arg:
                spec_index += 1
            kwargs_on_device[key] = device_arg

        return JaxMultichipWorkload(
            executable=multichip_workload.executable,
            compiled_executable=multichip_workload.compiled_executable,
            model=multichip_workload.model,
            args=args_on_device,
            kwargs=kwargs_on_device,
            device_mesh=multichip_workload.device_mesh,
            in_specs=multichip_workload.in_specs,
        )

    @staticmethod
    def _put_sharded_arg_on_multichip_device(
        arg: Any, device_mesh: Mesh, partition_spec: PartitionSpec | PyTree
    ) -> Any:
        """
        Puts workload argument on multichip device with proper sharding, depending on its type.
        """
        if isinstance(arg, Tensor):
            return jax.device_put(arg, NamedSharding(device_mesh, partition_spec))
        if isinstance(arg, PyTree):
            # TODO Assuming that only parameters are passed as PyTree for now. This will
            # work only for Flax linen parameters, revisit for other APIs.
            return jax.tree.map(
                lambda spec, param: jax.device_put(
                    param, NamedSharding(device_mesh, spec)
                ),
                partition_spec,
                arg,
                is_leaf=lambda x: (
                    isinstance(x, linen.Partitioned) or isinstance(x, PartitionSpec)
                ),
            )
        return arg
