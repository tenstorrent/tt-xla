# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev component bring-up tests."""

import pytest
import torch_xla.runtime as xr

from tests.runner.test_utils import get_xla_device_arch


def skip_on_wormhole(component: str) -> None:
    """Skip single-device FLUX.2 components on n150 runners.

    The nightly single_device job runs ``./tests/torch`` on both n150 (wormhole)
    and p150 (blackhole). FLUX.2 was only brought up / validated on Blackhole
    (single p150 and lb-blackhole), so skip the wormhole runner to avoid noise.
    """
    xr.set_device_type("TT")
    if get_xla_device_arch() == "wormhole":
        pytest.skip(
            f"FLUX.2 {component} validated on blackhole only; skipping n150 (wormhole)"
        )
