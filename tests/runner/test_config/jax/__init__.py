# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.infra.utilities.types import Framework

from ..config_loader import load_framework_test_configs

# Load JAX test configs using the centralized loader
_loaded = load_framework_test_configs(Framework.JAX)

# Empty placeholder models dictionary since all JAX models are discovered from tt-forge-models
PLACEHOLDER_MODELS = _loaded.get("PLACEHOLDER_MODELS", {})
test_config = _loaded.get("test_config", {})
