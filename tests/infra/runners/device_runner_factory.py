# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra.connectors import DeviceConnectorFactory
from infra.utilities import Framework

from .device_runner import DeviceRunner
from .jax_device_runner import JaxDeviceRunner
from .torch_device_runner import TorchDeviceRunner


class DeviceRunnerFactory:
    """Factory creating DeviceRunners based on provided framework."""

    @staticmethod
    def create_runner(framework: Framework) -> DeviceRunner:
        print(f"[DEBUG][DeviceRunnerFactory.create_runner] CALLED -- framework={framework}", flush=True)
        connector = DeviceConnectorFactory.create_connector(framework)
        print(f"[DEBUG][DeviceRunnerFactory.create_runner] Created connector: {type(connector).__name__}", flush=True)

        if framework == Framework.JAX:
            return JaxDeviceRunner(connector)
        elif framework == Framework.TORCH:
            runner = TorchDeviceRunner(connector)
            print(f"[DEBUG][DeviceRunnerFactory.create_runner] Created TorchDeviceRunner", flush=True)
            return runner
        else:
            raise ValueError(f"Unsupported framework {framework}")
