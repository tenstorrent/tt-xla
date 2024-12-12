# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Sequence

import jax

from .device_connector import DeviceType, connector
from .utils import Tensor


class DeviceRunner:
    @staticmethod
    def run_on_tt_device(f: Callable, inputs: Sequence[Tensor]) -> Tensor:
        """Runs test module on TT device."""
        return DeviceRunner._run_on_device(DeviceType.TT, f, inputs)

    @staticmethod
    def run_on_cpu(f: Callable, inputs: Sequence[Tensor]) -> Tensor:
        """Runs test module on CPU."""
        return DeviceRunner._run_on_device(DeviceType.CPU, f, inputs)

    @staticmethod
    def run_on_gpu(f: Callable, inputs: Sequence[Tensor]) -> Tensor:
        """Runs test module on GPU."""
        raise NotImplementedError("Support for GPUs not implemented")

    @staticmethod
    def put_on_tt_device(*tensors: Tensor) -> Sequence[Tensor]:
        """Puts `tensors` on TT device."""
        return DeviceRunner._put_on_device(DeviceType.TT, tensors)

    @staticmethod
    def put_on_cpu(*tensors: Tensor) -> Sequence[Tensor]:
        """Puts `tensors` on CPU."""
        return DeviceRunner._put_on_device(DeviceType.CPU, tensors)

    @staticmethod
    def put_on_gpu(*tensors: Tensor) -> Sequence[Tensor]:
        """Puts `tensors` on GPU."""
        raise NotImplementedError("Support for GPUs not implemented")

    @staticmethod
    def _run_on_device(
        device_type: DeviceType, f: Callable, inputs: Sequence[Tensor]
    ) -> Tensor:
        device = connector.connect_device(device_type)
        inputs = DeviceRunner._put_on_device(device_type, inputs)

        with jax.default_device(device):
            return f(*inputs)

    @staticmethod
    def _put_on_device(
        device_type: DeviceType, tensors: Sequence[Tensor]
    ) -> Sequence[Tensor]:
        """Puts `tensors` on device identified by `device_type`."""
        device = connector.connect_device(device_type)
        return [jax.device_put(t, device) for t in tensors]
