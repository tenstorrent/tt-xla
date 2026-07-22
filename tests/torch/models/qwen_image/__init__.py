# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image bring-up tests."""

import pytest
import torch_xla.runtime as xr

from tests.runner.test_utils import get_xla_device_arch


def skip_on_wormhole(component: str) -> None:
    """Skip on n150 (wormhole); Qwen-Image is validated on Blackhole only."""
    xr.set_device_type("TT")
    if get_xla_device_arch() == "wormhole":
        pytest.skip(
            f"Qwen-Image {component} validated on blackhole only; skipping n150 (wormhole)"
        )
