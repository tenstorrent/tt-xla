# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing reasons definition

from typing import Optional, Generator
from loguru import logger

from ..pytest import PyTestUtils

from .failing_reasons_common import ExceptionData, MessageChecker, MessageCheckerType

from .failing_reasons_xla import ExceptionCheck, FailingReason, FailingReasons


class FailingReasonsFinder:
    @classmethod
    def build_ex_data(cls, exception_value: Exception, exception_traceback: str) -> ExceptionData:
        """Convert exception to ExceptionData object

        Args:
            exception_value (Exception): Raised exception
            exception_traceback (str): Exception traceback

        Returns:
            ExceptionData: Exception data object
        """
        ex_class_name = f"{type(exception_value).__module__}.{type(exception_value).__name__}"
        ex_class_name = ex_class_name.replace("builtins.", "")
        ex_message = f"{exception_value}"
        exception_traceback = PyTestUtils.remove_colors(exception_traceback)
        ex_data = ExceptionData(
            class_name=ex_class_name,
            message=ex_message,
            error_log=exception_traceback,
        )
        return ex_data

    @classmethod
    def find_reason_by_ex_data(cls, ex: ExceptionData) -> Optional["FailingReasons"]:
        reasons = list(cls.find_reasons_by_ex_data(ex))
        if not reasons:
            return None
        if len(reasons) > 1:
            logger.warning(f"Multiple reasons found: {reasons} for ex: {ex}")
        return reasons[0]

    @classmethod
    def find_reasons_by_ex_data(cls, ex: ExceptionData) -> Generator["FailingReasons", None, None]:
        for failing_reason in FailingReasons:
            # Checking if exception data matches the failing reason
            if ex in failing_reason.value:
                yield failing_reason
