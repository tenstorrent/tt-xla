# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from enum import Enum

from loguru import logger


class LogLevel(Enum):
    """Enum representing valid Loguru levels"""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_env(cls, env_var="TTXLA_LOGGER_LEVEL", default="WARNING"):
        """Get level from environment variable with validation"""
        level_name = os.getenv(env_var, default).upper()

        try:
            logger.level(level_name)
            return cls(level_name)
        except ValueError:
            logger.warning(
                f"Invalid log level '{level_name}', using default '{default}'"
            )
            return cls(default)


_LOG_LEVEL = LogLevel.from_env()
logger.remove()  # clear sinks
logger.add(sys.stderr, level=_LOG_LEVEL.value)
