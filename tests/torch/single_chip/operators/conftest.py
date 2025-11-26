# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path


def adjust_python_path():
    print("Current sys.path:", sys.path)

    for src_path in [
        "third_party/tt_forge_sweeps/sweeps/src",
        # "../tt-forge-sweeps/sweeps/src",
    ]:
        src_path = Path(__file__).resolve().parents[4] / src_path
        if src_path.exists() and str(src_path) not in sys.path:
            # sys.path.insert(0, str(src_path))
            sys.path.append(str(src_path))

    print("Adjusted Python path for sweeps and forge.")
    print("Current sys.path:", sys.path)

adjust_python_path()


import _pytest
import _pytest.python
import _pytest.reports
import _pytest.runner
import pytest
from sweeps.utils import SweepsPytestReport


def pytest_generate_tests(metafunc):
    if "test_device" in metafunc.fixturenames:
        # Temporary work arround to provide dummy test_device
        # TODO remove workarround https://github.com/tenstorrent/tt-forge-fe/issues/342
        metafunc.parametrize("test_device", (None,), ids=["no_device"])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: _pytest.python.Function, call: _pytest.runner.CallInfo
):
    outcome = yield
    report: _pytest.reports.TestReport = outcome.get_result()
    SweepsPytestReport.adjust_report(item, call, outcome, report)
