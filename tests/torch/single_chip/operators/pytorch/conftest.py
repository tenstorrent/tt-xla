# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import _pytest
import _pytest.python
import _pytest.reports
import _pytest.runner

from ..utils.frontend import XLA_MODE

if XLA_MODE:
    import pluggy
else:
    import pluggy.callers

from loguru import logger

from ..utils import PyTestUtils
from ..utils import SweepsTagsLogger
from ..utils import FailingReasonsFinder
from ..utils import FailingReasonsValidation
from ..utils import TestPlanUtils


def pytest_generate_tests(metafunc):
    if "test_device" in metafunc.fixturenames:
        # Temporary work arround to provide dummy test_device
        # TODO remove workarround https://github.com/tenstorrent/tt-forge-fe/issues/342
        metafunc.parametrize("test_device", (None,), ids=["no_device"])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: _pytest.python.Function, call: _pytest.runner.CallInfo):
    if XLA_MODE:
        outcome: pluggy.Result = yield
    else:
        outcome: pluggy.callers._Result = yield
    report: _pytest.reports.TestReport = outcome.get_result()

    xfail_reason = None

    if report.when == "call" or (report.when == "setup" and report.skipped):
        xfail_reason = PyTestUtils.get_xfail_reason(item)

    # This hook function is called after each step of the test execution (setup, call, teardown)
    if call.when == "call":  # 'call' is a phase when the test is actually executed

        if xfail_reason is not None:  # an xfail reason is defined for the test
            SweepsTagsLogger.log_expected_failing_reason(xfail_reason=xfail_reason)

        if call.excinfo is not None:  # an exception occurred during the test execution

            logger.trace(
                f"Test: skipped: {report.skipped} failed: {report.failed} passed: {report.passed} report: {report}"
            )

            exception_value = call.excinfo.value
            long_repr = call.excinfo.getrepr(style="long")
            exception_traceback = str(long_repr)

            log_error_properties(item, exception_value, exception_traceback)

            ex_data = FailingReasonsFinder.build_ex_data(exception_value, exception_traceback)
            SweepsTagsLogger.log_detected_failing_reason(ex_data)

            if xfail_reason is not None:  # an xfail reason is defined for the test
                valid_reason = FailingReasonsValidation.validate_exception(
                    exception_value, exception_traceback, xfail_reason
                )

                # if reason is not valid, mark the test as failed and keep the original exception
                if valid_reason == False:
                    # Replace test report with a new one with outcome set to 'failed' and exception details
                    new_report = _pytest.reports.TestReport(
                        item=item,
                        when=call.when,
                        outcome="failed",
                        longrepr=call.excinfo.getrepr(style="long"),
                        sections=report.sections,
                        nodeid=report.nodeid,
                        location=report.location,
                        keywords=report.keywords,
                    )
                    outcome.force_result(new_report)
            else:
                logger.debug(f"Test '{item.name}' failed with exception: {type(exception_value)} '{exception_value}'")

    if report.when == "call" or (report.when == "setup" and report.skipped):
        try:
            log_test_vector_properties(
                item=item,
                report=report,
                xfail_reason=xfail_reason,
                exception=call.excinfo.value if call.excinfo is not None else None,
            )
        except Exception as e:
            logger.error(f"Failed to log test vector properties: {e}")
            logger.exception(e)


def log_error_properties(item: _pytest.python.Function, exception_value, exception_traceback):
    ex_class_name = f"{type(exception_value).__module__}.{type(exception_value).__name__}"
    ex_class_name = ex_class_name.replace("builtins.", "")
    item.user_properties.append(("exception_value", f"{ex_class_name}: {exception_value}"))
    item.user_properties.append(("exception_traceback", exception_traceback))


def log_test_vector_properties(
    item: _pytest.python.Function, report: _pytest.reports.TestReport, xfail_reason: str, exception: Exception
):
    original_name = item.originalname
    test_id = item.name
    test_id = test_id.replace(f"{original_name}[", "")
    test_id = test_id.replace("]", "")
    if test_id == "no_device-test_vector0":
        # This is not a valid test id. It happens when no tests are selected to run.
        return
    test_vector = TestPlanUtils.test_id_to_test_vector(test_id)

    SweepsTagsLogger.log_test_properties(test_vector=test_vector)

    item.user_properties.append(("id", test_id))
    item.user_properties.append(("operator", test_vector.operator))
    item.user_properties.append(
        ("input_source", test_vector.input_source.name if test_vector.input_source is not None else None)
    )
    item.user_properties.append(("dev_data_format", TestPlanUtils.dev_data_format_to_str(test_vector.dev_data_format)))
    item.user_properties.append(
        ("math_fidelity", test_vector.math_fidelity.name if test_vector.math_fidelity is not None else None)
    )
    item.user_properties.append(("input_shape", test_vector.input_shape))
    item.user_properties.append(("kwargs", test_vector.kwargs))
    if xfail_reason is not None:
        item.user_properties.append(("xfail_reason", xfail_reason))
    item.user_properties.append(("outcome", report.outcome))

    if exception is not None:
        error_message = f"{exception}"

        if "Observed maximum relative diff" in error_message:
            error_message_lines = error_message.split("\n")
            observed_error_lines = [line for line in error_message_lines if "Observed maximum relative diff" in line]
            if observed_error_lines:
                observed_error_line = observed_error_lines[0]
                # Example: "- Observed maximum relative diff: 0.0008770461427047849, maximum absolute diff: 0.0009063482284545898"
                rtol = float(observed_error_line.split(",")[0].split(":")[1].strip())
                atol = float(observed_error_line.split(",")[1].split(":")[1].strip())
            else:
                logger.error(f"Error parsing 'Observed maximum relative diff' from the exception: {error_message}")
                rtol = None
                atol = None
            item.user_properties.append(("all_close_rtol", rtol))
            item.user_properties.append(("all_close_atol", atol))
