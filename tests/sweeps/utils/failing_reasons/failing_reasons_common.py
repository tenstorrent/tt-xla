# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing reasons definition

import re

from dataclasses import dataclass
from typing import Callable


@dataclass
class ExceptionData:
    # operator: str
    class_name: str
    message: str
    error_log: str


MessageCheckerType = Callable[[str], bool]


class MessageChecker:
    @staticmethod
    def contains(message: str) -> bool:
        return lambda ex_message: message in ex_message

    @staticmethod
    def starts_with(message: str) -> bool:
        return lambda ex_message: ex_message.startswith(message)

    @staticmethod
    def equals(message: str) -> bool:
        return lambda ex_message: ex_message == message

    @staticmethod
    def regex(pattern: str) -> bool:
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
        return lambda ex_message: checker(ex_message.splitlines()[-1] if ex_message else ex_message)


M = MessageChecker
