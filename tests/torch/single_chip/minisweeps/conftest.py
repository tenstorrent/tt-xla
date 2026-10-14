# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import torch._dynamo
import torch_xla.runtime as xr
from loguru import logger


@pytest.fixture(autouse=True)
def run_around_tests():
    torch.manual_seed(0)
    yield
    torch._dynamo.reset()


@pytest.fixture(autouse=True)
def clear_torchxla_computation_cache():
    """
    Clears the TorchXLA computation cache after each test to prevent stale cached
    compilations from being served with wrong compile options when tests use different
    compiler configurations. See https://github.com/tenstorrent/tt-xla/issues/3439.
    """
    yield
    try:
        xr.clear_computation_cache()
    except Exception as e:
        logger.warning(f"Failed to clear TorchXLA computation cache: {e}")
        logger.warning(
            "This is expected if the test throws an exception, https://github.com/tenstorrent/tt-xla/issues/2814"
        )
