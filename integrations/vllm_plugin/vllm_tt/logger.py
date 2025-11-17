# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

import os

from vllm.logger import init_logger


def tt_init_logger(name: str):
    # Read log level from environment; default to "WARN" if not set.
    # Convert to uppercase for consistency.
    raw_level = os.getenv("LOGGER_LEVEL", "WARN").upper()

    # vLLM does not support "VERBOSE" level so treating it as "DEBUG" level.
    logger_level = {"VERBOSE": "DEBUG"}.get(raw_level, raw_level)

    if not logger_level in ["ERROR", "FATAL", "WARN", "WARNING", "INFO", "DEBUG"]:
        print(
            f"WARNING: Invalid logger_level={logger_level}; setting logger_level to 'WARN'."
        )
        logger_level = "WARN"

    # Create or retrieve a logger with the given name and set desired logging level.
    logger = init_logger(name)
    logger.setLevel(logger_level)

    return logger
