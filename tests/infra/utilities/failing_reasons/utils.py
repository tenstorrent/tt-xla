# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Utilities for failing reasons

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar, List, Optional

from loguru import logger
from pytest import ExceptionInfo

if TYPE_CHECKING:
    # ComponentChecker is only imported for type checking to avoid circular imports
    from .checks_xla_torch import ComponentChecker


@dataclass
class ExceptionData:
    class_name: str
    message: str
    error_log: str


MessageCheckerType = Callable[[str], bool]


class MessageChecker:
    """
    Class with helper methods to create message checker functions.
    Each method returns a function that takes an exception message as input
    and returns a boolean indicating whether the message matches the criteria.
    """

    @staticmethod
    def contains(message: str) -> bool:
        """Check if the message contains the given substring."""
        return lambda ex_message: message in ex_message

    @staticmethod
    def starts_with(message: str) -> bool:
        """Check if the message starts with the given substring."""
        return lambda ex_message: ex_message.startswith(message)

    @staticmethod
    def equals(message: str) -> bool:
        """Check if the message is equal to the given string."""
        return lambda ex_message: ex_message == message

    @staticmethod
    def regex(pattern: str) -> bool:
        """Check if the message matches the given regex pattern."""
        return lambda ex_message: re.search(pattern, ex_message) is not None

    @staticmethod
    def any(*checkers: MessageCheckerType) -> bool:
        """Check if any of the checkers match the message (or)."""
        return lambda ex_message: any(checker(ex_message) for checker in checkers)

    @staticmethod
    def neg(checker: MessageCheckerType) -> bool:
        """Negate the checker function (not)."""
        return lambda ex_message: not checker(ex_message)

    @staticmethod
    def last_line(checker: MessageCheckerType) -> str:
        """Apply the checker to the last line of the message."""
        return lambda ex_message: checker(
            ex_message.splitlines()[-1] if ex_message else ex_message
        )


# Short alias for MessageChecker used in failing reasons definitions
M = MessageChecker


@dataclass
class ExceptionCheck:
    """
    Class representing a set of checks to identify a specific exception.
    """

    class_name: Optional[str] = None
    component: Optional["ComponentChecker"] = None
    message: List[MessageCheckerType] = field(default_factory=list)
    error_log: List[MessageCheckerType] = field(default_factory=list)

    def __contains__(self, ex: ExceptionData) -> bool:
        """
        Check if the exception data matches this exception check via 'in' operator.
        """
        return self.check(ex)

    def check(self, ex: ExceptionData) -> bool:
        """
        Check if the exception data matches this exception check.

        Args:
            ex (ExceptionData): The exception data to check.

        Returns:
            bool: True if the exception data matches, False otherwise.
        """
        if self.class_name:
            if ex.class_name != self.class_name:
                return False
        if self.component is not None:
            if ex not in self.component:
                return False
        for message_check in self.message:
            if not message_check(ex.message):
                return False
        for message_check in self.error_log:
            if not message_check(ex.error_log):
                return False
        return True


@dataclass
class FailingReason:
    """
    Class representing a failing reason for a specific exception.
    It contains a description and a list of exception checks.
    """

    # Static class variable to be populated later to avoid circular import
    component_checker_none: ClassVar[Optional["ComponentChecker"]] = None

    description: str
    checks: List[ExceptionCheck] = field(default_factory=list)

    def __post_init__(self):
        self.checks = [
            check
            for check in self.checks
            if check.component is None
            or check.component != self.__class__.component_checker_none
        ]
        if len(self.checks) == 0:
            logger.trace(
                f"FailingReason '{self.description}' has no checks defined, it will not be used."
            )
        elif len(self.checks) > 1:
            logger.trace(
                f"FailingReason '{self.description}' has multiple ({len(self.checks)}) checks defined."
            )

    @property
    def component_checker(self) -> Optional["ComponentChecker"]:
        for check in self.checks:
            component = check.component
            if component is None or component == self.__class__.component_checker_none:
                continue
            return component
        return None

    @property
    def component_checker_description(self) -> Optional[str]:
        component_checker = self.component_checker
        return component_checker.description if component_checker else None

    def __contains__(self, ex: ExceptionData) -> bool:
        return self.check(ex)

    def check(self, ex: ExceptionData) -> bool:
        for check in self.checks:
            if ex in check:
                return True
        return False

    def __repr__(self) -> str:
        return f"FailingReason(description={self.description!r})"


class PyTestUtils:

    @classmethod
    def get_long_repr(cls, exc: Exception) -> str:
        """Get long representation of exception similar to pytest's longrepr."""
        long_repr = None
        if hasattr(exc, "__traceback__"):
            exc_info = (type(exc), exc, exc.__traceback__)
            long_repr = ExceptionInfo(exc_info).getrepr(style="long")
        else:
            long_repr = ExceptionInfo.from_exc_info().getrepr(style="long")
        long_repr = str(long_repr)
        return long_repr

    @classmethod
    def remove_colors(cls, text: str) -> str:
        # Remove colors from text
        text = re.sub(r"#x1B\[\d+m", "", text)
        text = re.sub(r"#x1B\[\d+;\d+;\d+m", "", text)
        text = re.sub(r"#x1B\[\d+;\d+;\d+;\d+;\d+m", "", text)

        text = re.sub(r"\[\d+m", "", text)
        text = re.sub(r"\[\d+;\d+;\d+m", "", text)
        text = re.sub(r"\[\d+;\d+;\d+;\d+;\d+m", "", text)

        text = re.sub(r"\[1A", "", text)
        text = re.sub(r"\[1B", "", text)
        text = re.sub(r"\[2K", "", text)

        return text
