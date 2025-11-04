# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Utilities for failing reasons

import re
from dataclasses import dataclass
from typing import Callable


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
