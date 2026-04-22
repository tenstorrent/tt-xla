# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .device_connector import DeviceConnector, DeviceType
from .device_connector_factory import DeviceConnectorFactory
from .torch_device_connector import TorchDeviceConnector, torch_device_connector

# Keep JAX connector import lazy so torch/CUDA-only lanes do not require the JAX stack.
JaxDeviceConnector = None
jax_device_connector = None
