# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Detect xla mode

import os

from loguru import logger


class FrontendDetector:
    def get_xla_mode() -> bool:
        module_file_path = os.path.abspath(__file__)
        if "tt-xla" in module_file_path:
            return True
        print(f"module_file_path: {module_file_path}")
        return False

    XLA_MODE = get_xla_mode()


# FrontendDetector.XLA_MODE = False
# FrontendDetector.XLA_MODE = True

XLA_MODE = FrontendDetector.XLA_MODE
logger.info(f"XLA_MODE: {XLA_MODE}")

__all__ = ["XLA_MODE"]
