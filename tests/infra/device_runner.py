# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Callable, Sequence

import jax
from jax.sharding import NamedSharding

from .device_connector import DeviceType, device_connector
from .multichip_utils import ShardingMode
from .types import Tensor
from .workload import MultichipWorkload, Workload


class DeviceRunner:
    """
    Class providing methods to put and run workload on any supported device.
    """

    # ---------- Public methods ----------

    @staticmethod
    def run_on_tt_device(workload: Workload, device_num: int = 0) -> Tensor:
        """Runs `workload` on TT device."""
        return DeviceRunner._run_on_device(workload, DeviceType.TT, device_num)

    @staticmethod
    def run_on_multichip_device(
        multichip_workload: MultichipWorkload, sharding_mode: ShardingMode
    ) -> Tensor:
        """
        Runs `workload` on a multichip device.
        Depending on the sharding mode, we might put the workload directly on device, ot leave
        it to the jax to infer on the fly.
        """
        if sharding_mode.requires_device_put:
            sharded_workload = DeviceRunner._put_multichip_workload_on_device(
                multichip_workload
            )
            return sharded_workload.execute()
        else:
            return multichip_workload.execute()

    @staticmethod
    def run_on_cpu(workload: Workload) -> Tensor:
        """Runs `workload` on CPU."""
        return DeviceRunner._run_on_device(workload, DeviceType.CPU)

    @staticmethod
    def run_on_gpu(workload: Workload) -> Tensor:
        """Runs `workload` on GPU."""
        raise NotImplementedError("Support for GPUs not implemented")

    @staticmethod
    def put_on_tt_device(workload: Workload, device_num: int = 0) -> Workload:
        """Puts `workload` on TT device."""
        return DeviceRunner._put_on_device(workload, DeviceType.TT, device_num)

    @staticmethod
    def put_on_cpu(workload: Workload) -> Workload:
        """Puts `workload` on CPU."""
        return DeviceRunner._put_on_device(workload, DeviceType.CPU)

    @staticmethod
    def put_on_gpu(workload: Workload) -> Workload:
        """Puts `workload` on GPU."""
        raise NotImplementedError("Support for GPUs not implemented")

    @staticmethod
    def put_tensors_on_tt_device(*tensors: Tensor) -> Sequence[Tensor]:
        """Puts `tensors` on TT device."""
        return DeviceRunner._put_tensors_on_device(DeviceType.TT, tensors)

    @staticmethod
    def put_tensors_on_cpu(*tensors: Tensor) -> Sequence[Tensor]:
        """Puts `tensors` on CPU."""
        return DeviceRunner._put_tensors_on_device(DeviceType.CPU, tensors)

    @staticmethod
    def put_tensors_on_gpu(*tensors: Tensor) -> Sequence[Tensor]:
        """Puts `tensors` on GPU."""
        raise NotImplementedError("Support for GPUs not implemented")

    # ---------- Private methods ----------

    @staticmethod
    def _run_on_device(
        workload: Workload, device_type: DeviceType, device_num: int = 0
    ) -> Tensor:
        """Runs `workload` on device identified by `device_type`."""
        device_workload = DeviceRunner._put_on_device(workload, device_type, device_num)
        device = device_connector.connect_device(device_type, device_num)

        with jax.default_device(device):
            return device_workload.execute()

    @staticmethod
    def _put_sharded_tensor_on_multichip_device(
        tensor: Tensor, mesh: jax.sharding.Mesh, in_spec: jax.sharding.PartitionSpec
    ) -> Tensor:
        """
        Needed for multichip: Uses put_device to give inputs shardings corresponding to the ones in
        shard_map() function.
        """
        return jax.device_put(tensor, NamedSharding(mesh, in_spec), may_alias=True)

    @staticmethod
    def _put_on_device(
        workload: Workload, device_type: DeviceType, device_num: int = 0
    ) -> Workload:
        """Puts `workload` on device and returns it."""
        device = device_connector.connect_device(device_type, device_num)
        return DeviceRunner._safely_put_workload_on_device(workload, device)

    @staticmethod
    def _put_multichip_workload_on_device(
        multichip_workload: MultichipWorkload,
    ) -> MultichipWorkload:
        """Gives the workload inputs shardings, necessary for multichip workloads"""
        args_on_device = []
        spec_index = 0
        # TODO: It might necessary to put a try-except block here, but holding that off until we
        # come across a case where it's needed.
        for arg in multichip_workload.args:
            if not isinstance(arg, Tensor):
                args_on_device.append(arg)
            else:
                args_on_device.append(
                    DeviceRunner._put_sharded_tensor_on_multichip_device(
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
                kwargs_on_device[
                    key
                ] = DeviceRunner._put_sharded_tensor_on_multichip_device(
                    value,
                    multichip_workload.device_mesh,
                    multichip_workload.in_specs[spec_index],
                )
                spec_index += 1

        return MultichipWorkload(
            multichip_workload.executable,
            args_on_device,
            kwargs_on_device,
            device_mesh=multichip_workload.device_mesh,
            in_specs=multichip_workload.in_specs,
        )

    @staticmethod
    def _put_tensors_on_device(
        device_type: DeviceType, tensors: Sequence[Tensor]
    ) -> Sequence[Tensor]:
        """Puts `tensors` on device identified by `device_type`."""
        device = device_connector.connect_device(device_type)
        return [jax.device_put(t, device) for t in tensors]

    @staticmethod
    def _safely_put_workload_on_device(
        workload: Workload, device: jax.Device
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
        fn_params = list(inspect.signature(workload.executable).parameters.keys())

        args_on_device = []
        for i, arg in enumerate(workload.args):
            if fn_params[i] not in workload.static_argnames:
                try:
                    args_on_device.append(jax.device_put(arg, device))
                except:
                    args_on_device.append(arg)
            else:
                args_on_device.append(arg)

        kwargs_on_device = {}
        for key, value in workload.kwargs.items():
            if key not in workload.static_argnames:
                try:
                    kwargs_on_device[key] = jax.device_put(value, device)
                except:
                    kwargs_on_device[key] = value
            else:
                kwargs_on_device[key] = value

        return Workload(workload.executable, args_on_device, kwargs_on_device)


# --------------- Convenience decorators ---------------


def run_on_tt_device(f: Callable):
    """Runs any decorated function `f` on TT device."""

    def wrapper(*args, **kwargs):
        workload = Workload(f, args, kwargs)
        return DeviceRunner.run_on_tt_device(workload)

    return wrapper


def run_on_cpu(f: Callable):
    """Runs any decorated function `f` on CPU."""

    def wrapper(*args, **kwargs):
        workload = Workload(f, args, kwargs)
        return DeviceRunner.run_on_cpu(workload)

    return wrapper


def run_on_gpu(f: Callable):
    """Runs any decorated function `f` on GPU."""

    def wrapper(*args, **kwargs):
        workload = Workload(f, args, kwargs)
        return DeviceRunner.run_on_gpu(workload)

    return wrapper
