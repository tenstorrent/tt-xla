# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Sequence

import jax
from connectors import DeviceConnector, DeviceType
from jax.sharding import NamedSharding
from utilities.multichip_utils import MultichipWorkload, ShardingMode
from utilities.types import Device, Tensor
from utilities.workloads.jax_workload import JaxWorkload, Workload

from .device_runner import DeviceRunner


class JaxDeviceRunner(DeviceRunner):
    """Device runner used with JAX."""

    # -------------------- Public methods --------------------

    def __init__(self, connector: DeviceConnector) -> None:
        super().__init__(connector)

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _run_on_device(
        self, workload: Workload, device_type: DeviceType, device_num: int = 0
    ) -> Tensor:
        device = self._device_connector.connect_device(device_type, device_num)
        device_workload = self._put_on_device(workload, device=device)

        with jax.default_device(device):
            return device_workload.execute()

    # @override
    def _run_on_multichip_device(
        self, multichip_workload: MultichipWorkload, sharding_mode: ShardingMode
    ) -> Tensor:
        """
        Runs `workload` on a multichip device.

        Depending on the sharding mode, we might put the workload directly on device, or
        leave it to jax to infer on the fly.
        """
        if sharding_mode.requires_device_put:
            sharded_workload = self._put_multichip_workload_on_device(
                multichip_workload
            )
            return sharded_workload.execute()
        else:
            return multichip_workload.execute()

    # @override
    def _put_tensors_on_device(
        self, device_type: DeviceType, tensors: Sequence[Tensor]
    ) -> Sequence[Tensor]:
        device = self._device_connector.connect_device(device_type)
        return [jax.device_put(t, device) for t in tensors]

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
        assert isinstance(workload, JaxWorkload)

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

        return JaxWorkload(
            workload.executable,  # Unchanged.
            args_on_device,
            kwargs_on_device,
            workload.static_argnames,  # Unchanged.
        )

    # -----------------

    def _put_multichip_workload_on_device(
        self,
        multichip_workload: MultichipWorkload,
    ) -> MultichipWorkload:
        """Gives the workload inputs shardings, necessary for multichip workloads."""
        args_on_device = []
        spec_index = 0
        # TODO: It might necessary to put a try-except block here, but holding that off
        # until we come across a case where it's needed.
        for arg in multichip_workload.args:
            if not isinstance(arg, Tensor):
                args_on_device.append(arg)
            else:
                args_on_device.append(
                    self._put_sharded_tensor_on_multichip_device(
                        arg,
                        multichip_workload.device_mesh,
                        multichip_workload.in_specs[spec_index],
                    )
                )
                spec_index += 1

        kwargs_on_device = {}
        for key, value in multichip_workload.kwargs.items():
            if not isinstance(value, Tensor):
                kwargs_on_device[key] = value
            else:
                kwargs_on_device[key] = self._put_sharded_tensor_on_multichip_device(
                    value,
                    multichip_workload.device_mesh,
                    multichip_workload.in_specs[spec_index],
                )
                spec_index += 1

        return MultichipWorkload(
            multichip_workload.executable,
            args_on_device,
            kwargs_on_device,
            multichip_workload.static_argnames,
            multichip_workload.device_mesh,
            multichip_workload.in_specs,
        )

    def _put_sharded_tensor_on_multichip_device(
        self,
        tensor: Tensor,
        mesh: jax.sharding.Mesh,
        in_spec: jax.sharding.PartitionSpec,
    ) -> Tensor:
        """
        Uses `device_put` to give inputs shardings corresponding to the ones in
        shard_map() function.
        """
        return jax.device_put(tensor, NamedSharding(mesh, in_spec), may_alias=True)
