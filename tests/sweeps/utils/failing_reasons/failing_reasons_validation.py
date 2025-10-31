# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Validations of failing reasons


from typing import Optional

from loguru import logger

from .failing_reasons import ExceptionData, FailingReasons, FailingReasonsFinder


class FailingReasonsValidation:
    @classmethod
    def validate_exception(
        cls, exception_value: Exception, exception_traceback: str, xfail_reason: str
    ) -> Optional[bool]:
        """Validate exception based on xfail reason

        Args:
            exception_value (Exception): Raised exception to validate
            xfail_reason (str): Xfail reason

        Returns:
            bool: True if exception message and type match the expected values, False otherwise, None if no check is defined
        """
        failing_reason = FailingReasons.find_by_description(xfail_reason)
        if failing_reason is None:
            logger.error(
                f"Test is marked with xfail reason: '{xfail_reason}' but no failing reason found for exception: {type(exception_value)} '{exception_value}'"
            )
            return False

        if len(failing_reason.value.checks) > 0:
            ex_data = FailingReasonsFinder.build_ex_data(
                exception_value, exception_traceback
            )

            # Checking if exception data matches the failing reason
            if ex_data in failing_reason.value:
                logger.trace(
                    f"Correct xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'"
                )
                return True
            logger.error(
                f"Wrong xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'"
            )
            logger.debug(f"Exception data: {ex_data} failing reason: {failing_reason}")
            return False
        else:
            logger.warning(
                f"Test is marked with xfail reason: '{xfail_reason}' but no check performed for exception: {type(exception_value)} '{exception_value}'"
            )
            return None
