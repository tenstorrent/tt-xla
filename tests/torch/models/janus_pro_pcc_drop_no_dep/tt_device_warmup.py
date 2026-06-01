# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Warm up TT device firmware JIT before op tests (cold-cache parallel build race)."""

from __future__ import annotations

import time

import torch_xla
import torch_xla.runtime as xr

from infra.connectors import DeviceConnectorFactory
from infra.utilities import Framework
from tests.infra.testers.compiler_config import CompilerConfig

_JIT_DIR_RACE = "cannot create directories"


def ensure_tt_device_ready(*, max_attempts: int = 5, retry_delay_s: float = 2.0) -> None:
    """
    Initialize TT runtime and block until firmware JIT finishes.

    On a cold ``tt-metal-cache``, ``set_custom_compile_options`` can trigger parallel
    firmware builds (``cq_dispatch*``) that race on ``fs::create_directories`` and fail
    with ``File exists``. Retrying after a short delay matches what happens implicitly
    when Janus weights load slowly after ``xr.set_device_type("TT")``.
    """
    xr.set_device_type("TT")
    DeviceConnectorFactory.create_connector(Framework.TORCH)

    options = CompilerConfig().to_torch_compile_options()
    last_error: RuntimeError | None = None
    for attempt in range(max_attempts):
        try:
            torch_xla.set_custom_compile_options(options)
            return
        except RuntimeError as exc:
            last_error = exc
            if _JIT_DIR_RACE not in str(exc) or attempt + 1 >= max_attempts:
                raise
            time.sleep(retry_delay_s * (attempt + 1))

    if last_error is not None:
        raise last_error
