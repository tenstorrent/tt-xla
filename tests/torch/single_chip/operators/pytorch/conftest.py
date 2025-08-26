# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import _pytest
import _pytest.python
import _pytest.reports
import _pytest.runner

from ..utils import SweepsPytestReport


def pytest_generate_tests(metafunc):
    if "test_device" in metafunc.fixturenames:
        # Temporary work arround to provide dummy test_device
        # TODO remove workarround https://github.com/tenstorrent/tt-forge-fe/issues/342
        metafunc.parametrize("test_device", (None,), ids=["no_device"])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: _pytest.python.Function, call: _pytest.runner.CallInfo):
    outcome = yield
    report: _pytest.reports.TestReport = outcome.get_result()
    SweepsPytestReport.adjust_report(item, call, outcome, report)
