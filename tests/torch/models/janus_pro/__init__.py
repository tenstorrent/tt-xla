# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Janus-Pro T2I component bring-up tests."""

import pytest
import torch_xla.runtime as xr

from tests.runner.test_utils import get_xla_device_arch


def skip_pro_7b_image_token_on_wormhole() -> None:
    """Skip Pro-7B ImageTokenStep on n150 runners (DRAM OOM on wormhole)."""
    xr.set_device_type("TT")
    if get_xla_device_arch() == "wormhole":
        pytest.skip(
            "Pro-7B ImageTokenStep OOM on n150 (wormhole); requires p150 (blackhole)"
        )
