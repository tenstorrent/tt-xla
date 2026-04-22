# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal runner exports.

Avoid eager imports of backend-specific runners so torch/CUDA-only collection does
not recursively pull TT/XLA and JAX machinery through package barrels.
"""

from .utils import run_on_cpu, run_on_cuda_device, run_on_tt_device

try:
    from .device_runner import DeviceRunner
except Exception:
    DeviceRunner = None

try:
    from .device_runner_factory import DeviceRunnerFactory
except Exception:
    DeviceRunnerFactory = None

try:
    from .jax_device_runner import JaxDeviceRunner
except Exception:
    JaxDeviceRunner = None

try:
    from .torch_device_runner import TorchDeviceRunner, to_device
except Exception:
    TorchDeviceRunner = None
    to_device = None
