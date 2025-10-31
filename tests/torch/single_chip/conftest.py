# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import _pytest
import _pytest.python
import _pytest.reports
import _pytest.runner
import pluggy
import pytest
from loguru import logger
from sweeps.utils.failing_reasons import (
    ExceptionData,
    FailingReasons,
    FailingReasonsFinder,
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: _pytest.python.Function, call: _pytest.runner.CallInfo
):
    outcome: pluggy.Result = yield
    report: _pytest.reports.TestReport = outcome.get_result()

    # xfail_reason = None

    # if report.when == "call" or (report.when == "setup" and report.skipped):
    #     xfail_reason = PyTestUtils.get_xfail_reason(item)

    # This hook function is called after each step of the test execution (setup, call, teardown)
    if call.when == "call":  # 'call' is a phase when the test is actually executed

        # if xfail_reason is not None:  # an xfail reason is defined for the test
        #     SweepsTagsLogger.log_expected_failing_reason(xfail_reason=xfail_reason)

        # Only process if the test has failed
        if call.excinfo is not None:  # an exception occurred during the test execution

            # logger.trace(
            #     f"Test: skipped: {report.skipped} failed: {report.failed} passed: {report.passed} report: {report}"
            # )

            # Extract exception information from the call info
            exception_value = call.excinfo.value
            # Get the long representation of the exception
            long_repr = call.excinfo.getrepr(style="long")
            # Convert the long representation to string
            exception_traceback = str(long_repr)

            # cls.log_error_properties(item, exception_value, exception_traceback)

            # Build ExceptionData from exception
            ex_data: ExceptionData = FailingReasonsFinder.build_ex_data(
                exception_value, exception_traceback
            )
            # Find failing reason by ExceptionData
            failing_reason = FailingReasonsFinder.find_reason_by_ex_data(ex_data)

            # If no failing reason is found, classify as UNCLASSIFIED
            if not failing_reason:
                failing_reason = FailingReasons.UNCLASSIFIED

            # Log detected failing reason
            logger.warning(
                f"Detected failing reason: {failing_reason.name} - {failing_reason.value.description} in component: {failing_reason.value.component_checker_description}"
            )
