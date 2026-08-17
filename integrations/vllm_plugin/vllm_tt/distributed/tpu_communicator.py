# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

"""Thin wrapper for TPU device communication.

Uses vLLM's base device communicator as the underlying implementation.
"""

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


class TpuCommunicator(DeviceCommunicatorBase):
    """TPU device communicator for distributed execution on Tenstorrent devices."""

    pass
