# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# pytest utilities

import re
import sys

import pytest
from loguru import logger


class PyTestUtils:

    @classmethod
    def get_long_repr(cls, exc: Exception) -> str:
        """Get long representation of exception similar to pytest's longrepr."""
        long_repr = None
        if hasattr(exc, "__traceback__"):
            exc_info = (type(exc), exc, exc.__traceback__)
            long_repr = pytest.ExceptionInfo(exc_info).getrepr(style="long")
        else:
            long_repr = pytest.ExceptionInfo.from_exc_info().getrepr(style="long")
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
