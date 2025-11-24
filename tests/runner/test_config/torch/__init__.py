# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.infra.utilities.types import Framework

from ..config_loader import load_framework_test_configs

# Load torch test configs using the centralized loader
_loaded = load_framework_test_configs(Framework.TORCH)

PLACEHOLDER_MODELS = _loaded.get("PLACEHOLDER_MODELS", {})
test_config = _loaded.get("test_config", {})
