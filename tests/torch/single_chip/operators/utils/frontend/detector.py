# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Detect xla mode

import os

from loguru import logger
from dataclasses import dataclass


@dataclass
class ProjectSelectorParams:
    """Test features configuration parameters"""

    active_project: str = os.getenv("ACTIVATE_PROJECT", "sweeps")
    active_frontend: str = os.getenv("ACTIVATE_FRONTEND", "forge")


params = ProjectSelectorParams()


class FrontendDetector:
    @staticmethod
    def get_xla_mode_via_path() -> bool:
        module_file_path = os.path.abspath(__file__)
        if "tt-xla" in module_file_path:
            return True
        print(f"module_file_path: {module_file_path}")
        return False

    @staticmethod
    def get_xla_mode_via_env_var() -> bool:
        logger.info(f"params.active_frontend: {params.active_frontend}")
        return params.active_frontend.lower() == "xla"

    # XLA_MODE = get_xla_mode_via_path()
    XLA_MODE = get_xla_mode_via_env_var()


# FrontendDetector.XLA_MODE = False
# FrontendDetector.XLA_MODE = True

XLA_MODE = FrontendDetector.XLA_MODE
logger.info(f"XLA_MODE: {XLA_MODE}")

__all__ = ["XLA_MODE"]
