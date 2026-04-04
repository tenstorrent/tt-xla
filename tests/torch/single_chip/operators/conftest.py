# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import _pytest
import _pytest.python
import _pytest.reports
import _pytest.runner
import pytest
import torch
import torch._dynamo
import torch_xla.runtime as xr
from loguru import logger
from sweeps.core.logging import SweepsPytestReport
from sweeps.core.logging.sweeps_property_utils import (
    ForgePropertyHandler,
    forge_property_handler_var,
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: _pytest.python.Function, call: _pytest.runner.CallInfo
):
    outcome = yield
    report: _pytest.reports.TestReport = outcome.get_result()
    SweepsPytestReport.adjust_report(item, call, outcome, report)


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


@pytest.fixture(scope="function", autouse=True)
def forge_property_recorder(request, record_property):
    # Create a handler that uses the property store; the handler is responsible for recording and managing property details.
    forge_property_handler = ForgePropertyHandler()
    token = forge_property_handler_var.set(forge_property_handler)

    yield

    forge_property_handler.record_error(request)
    forge_property_handler.record_all_properties(record_property)
    forge_property_handler_var.reset(token)
