# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Callable


def fullname(fn: Callable):
    """Returns informative name for a function."""
    module = fn.__module__

    if module == "builtins":
        return fn.__name__  # avoid outputs like 'builtins.str'

    return module + "::" + fn.__name__


class Colors(Enum):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


color_green = lambda string: f"{Colors.OKGREEN.value}{string}{Colors.ENDC.value}"
color_red = lambda string: f"{Colors.FAIL.value}{string}{Colors.ENDC.value}"

PASSED = color_green("PASSED")
FAILED = color_red("FAILED")

passed = lambda string: print(f"{string} {PASSED}")
failed = lambda string: print(f"{string} {FAILED}")
