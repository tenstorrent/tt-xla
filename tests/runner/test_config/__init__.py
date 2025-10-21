# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .config_loader import load_all_test_configs

# Load all test config objects from yaml files and expose as python objects.
_loaded = load_all_test_configs()

PLACEHOLDER_MODELS = _loaded.get("PLACEHOLDER_MODELS", {})
test_config = _loaded.get("test_config", {})

assert isinstance(
    PLACEHOLDER_MODELS, dict
), f"Expected dict for PLACEHOLDER_MODELS, got {type(PLACEHOLDER_MODELS).__name__}"
assert isinstance(
    test_config, dict
), f"Expected dict for test_config, got {type(test_config).__name__}"
