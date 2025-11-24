# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing reasons finder

from typing import Generator, Optional

from loguru import logger

from .checks_xla import FailingReasons
from .utils import ExceptionData, PyTestUtils


class FailingReasonsFinder:
    @classmethod
    def build_ex_data(
        cls,
        exception_value: Exception,
        exception_traceback: str,
        stdout: str = None,
        stderr: str = None,
    ) -> ExceptionData:
        """Convert exception to ExceptionData object

        Args:
            exception_value (Exception): Raised exception
            exception_traceback (str): Exception traceback

        Returns:
            ExceptionData: Exception data object
        """
        ex_class_name = (
            f"{type(exception_value).__module__}.{type(exception_value).__name__}"
        )
        ex_class_name = ex_class_name.replace("builtins.", "")
        ex_message = f"{exception_value}"
        exception_traceback = PyTestUtils.remove_colors(exception_traceback)
        ex_data = ExceptionData(
            class_name=ex_class_name,
            message=ex_message,
            error_log=exception_traceback,
            stdout=stdout,
            stderr=stderr,
        )
        return ex_data

    @classmethod
    def find_reason_by_ex_data(cls, ex: ExceptionData) -> Optional["FailingReasons"]:
        reasons = list(cls.find_reasons_by_ex_data(ex))
        if not reasons:
            # If no failing reason is found, classify as UNCLASSIFIED
            return FailingReasons.UNCLASSIFIED
        if len(reasons) > 1:
            logger.warning(f"Multiple reasons found: {reasons} for ex: {ex}")
        return reasons[0]

    @classmethod
    def find_reasons_by_ex_data(
        cls, ex: ExceptionData
    ) -> Generator["FailingReasons", None, None]:
        for failing_reason in FailingReasons:
            # Checking if exception data matches the failing reason
            if ex in failing_reason.value:
                yield failing_reason

    @classmethod
    def find_reason_by_exception(
        cls, exc: Exception, stdout: str, stderr: str
    ) -> Optional["FailingReasons"]:
        """Find failing reason by exception"""
        # Get long representation of exception
        long_repr = PyTestUtils.get_long_repr(exc)

        # Build ExceptionData from exception
        ex_data: ExceptionData = FailingReasonsFinder.build_ex_data(
            exc, long_repr, stdout, stderr
        )

        # Find failing reason by ExceptionData
        failing_reason = FailingReasonsFinder.find_reason_by_ex_data(ex_data)

        return failing_reason
