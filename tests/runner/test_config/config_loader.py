# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict

import yaml

from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus

# Single source of truth for the placeholders YAML filename
PLACEHOLDERS_FILENAME = "test_config_placeholders.yaml"


# Map YAML 'status' string (enum name or value) to ModelTestStatus; raises on invalid values if provided.
def _enum_map_status(value: str):
    if not value:
        raise ValueError("Missing 'status' field or empty value")
    value = str(value).strip()
    try:
        return ModelTestStatus[value.upper()]  # Try enum name
    except KeyError:
        try:
            return ModelTestStatus(value)  # Try enum value
        except ValueError:
            raise ValueError(f"Invalid ModelTestStatus: '{value}'")


# Map YAML 'bringup_status' string (enum name or value) to BringupStatus; raises on invalid values if provided.
def _enum_map_bringup(value: str):
    # Field may be omitted entirely; only validate if present
    if value is None:
        return None
    value = str(value).strip()
    try:
        return BringupStatus[value.upper()]  # Try enum name
    except KeyError:
        try:
            return BringupStatus(value)  # Try enum value
        except ValueError:
            raise ValueError(f"Invalid BringupStatus: '{value}'")


# Normalize a config dict (status/bringup_status in top level or nested under arch_overrides).
def _normalize_enums_in_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg or {})

    if not cfg:
        return {}

    # Mapping of YAML keys to conversion functions
    enum_fields = {
        "status": _enum_map_status,
        "bringup_status": _enum_map_bringup,
    }

    # Apply normalization for any known enum field
    for key, mapper in enum_fields.items():
        if key in cfg:
            cfg[key] = mapper(cfg[key])  # type: ignore

    # Recursively normalize arch_overrides, if present
    overrides = cfg.get("arch_overrides")
    if isinstance(overrides, dict):
        for arch, arch_cfg in overrides.items():
            if isinstance(arch_cfg, dict):
                overrides[arch] = _normalize_enums_in_cfg(arch_cfg)

    return cfg


# Load placeholders from dedicated YAML and merge test entries from other YAMLs.
def load_all_test_configs() -> Dict[str, Any]:
    """Load and merge YAML-based test configurations for model bring-up and CI.

    Files:
    - test_config_placeholders.yaml: defines PLACEHOLDER_MODELS entries.
    - test_config_*.yaml: define test_config entries grouped by mode (e.g. inference, training).

    Returns:
    dict: {
        "PLACEHOLDER_MODELS": <dict of placeholder configs>,
        "test_config": <merged dict of all test configs>
    }
    """
    config_dir = Path(__file__).parent

    # Load placeholders only from the dedicated file
    merged_placeholders: Dict[str, Any] = {}
    placeholders_file = config_dir / PLACEHOLDERS_FILENAME
    if placeholders_file.exists():
        with open(placeholders_file, "r") as fh:
            data = yaml.safe_load(fh) or {}
        placeholders = data.get("PLACEHOLDER_MODELS", {}) or {}
        for k, v in placeholders.items():
            entry = dict(v or {})
            merged_placeholders[k] = _normalize_enums_in_cfg(entry)

    # Merge test_config from all other YAMLs
    merged_test_config: Dict[str, Any] = {}
    for yaml_file in sorted(config_dir.glob("test_config_*.yaml")):
        if yaml_file.name == PLACEHOLDERS_FILENAME:
            continue
        with open(yaml_file, "r") as fh:
            data = yaml.safe_load(fh) or {}
        tests = data.get("test_config", {}) or {}
        for test_id, cfg in tests.items():
            merged_test_config[test_id] = _normalize_enums_in_cfg(cfg)

    return {
        "PLACEHOLDER_MODELS": merged_placeholders,
        "test_config": merged_test_config,
    }
