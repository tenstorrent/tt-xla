# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger
from dataclasses import dataclass


@dataclass
class TestSweepsFeaturesParams:
    """Test features configuration parameters"""

    __test__ = False  # Disable pytest collection

    skip_forge_verification: bool
    dry_run: bool
    ignore_xfail_files: bool
    ignore_skip_files: bool

    @staticmethod
    def get_env_property(env_var: str, default_value: str):
        return os.getenv(env_var, default_value)

    @staticmethod
    def get_env_property_bool(env_var: str, default_value: bool):
        return os.getenv(env_var, f"{default_value}".lower()).lower() == "true"

    @classmethod
    def from_env(cls):
        """Create a TestSweepsFeaturesParams object from environment variables"""

        skip_forge_verification = cls.get_env_property_bool("SKIP_FORGE_VERIFICATION", False)
        dry_run = cls.get_env_property_bool("DRY_RUN", False)
        ignore_xfail_files = cls.get_env_property_bool("IGNORE_XFAIL_FILES", False)
        ignore_skip_files = cls.get_env_property_bool("IGNORE_SKIP_FILES", False)

        # Construct feature parameters
        feature_params = cls(
            skip_forge_verification=skip_forge_verification,
            dry_run=dry_run,
            ignore_xfail_files=ignore_xfail_files,
            ignore_skip_files=ignore_skip_files,
        )

        logger.info(f"Features parameters: {feature_params}")

        return feature_params


class TestSweepsFeatures:
    """Store test features configuration parameters"""

    __test__ = False  # Disable pytest collection

    params = TestSweepsFeaturesParams.from_env()
