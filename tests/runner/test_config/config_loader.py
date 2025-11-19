# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict

from ruamel.yaml import YAML

from tests.infra.utilities.types import Framework
from tests.runner.test_utils import ModelTestStatus
from tests.utils import BringupStatus

# Single source of truth for the placeholders YAML filename
PLACEHOLDERS_FILENAME = "test_config_placeholders.yaml"

# Path to filecheck pattern files
FILECHECK_DIR = Path(__file__).parent.parent.parent / "filecheck"


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


# Define allowed fields, used for validation of test config YAML to catch unknown fields / typos.
_ALLOWED_FIELDS = {
    # Comparator controls
    "required_pcc",
    "assert_pcc",
    "assert_atol",
    "required_atol",
    "assert_allclose",
    "allclose_rtol",
    "allclose_atol",
    # Status/metadata
    "status",
    "reason",
    "bringup_status",
    "markers",
    "supported_archs",
    "batch_size",
    # Nested arch overrides
    "arch_overrides",
    # Needed for training tests
    "execution_pass",
    # FileCheck patterns list
    "filechecks",
}


# Validate top-level and nested fields to ensure they are from the allowed fields list.
def _validate_allowed_keys(cfg: Dict[str, Any], *, ctx: str) -> None:
    def validate_mapping(mapping: Dict[str, Any], where: str) -> None:
        for k in mapping.keys():
            if k not in _ALLOWED_FIELDS:
                raise ValueError(
                    f"Unknown field '{k}' in {where} of {ctx}. Allowed: {sorted(_ALLOWED_FIELDS)}"
                )

    validate_mapping(cfg, where="entry")

    overrides = cfg.get("arch_overrides")
    if isinstance(overrides, dict):
        for arch, arch_cfg in overrides.items():
            if isinstance(arch_cfg, dict):
                validate_mapping(arch_cfg, where=f"arch_overrides['{arch}']")


# Validate that filecheck pattern files referenced in config exist in tests/filecheck/
def _validate_filecheck_references(
    cfg: Dict[str, Any], *, test_id: str, yaml_file: str
) -> None:
    """Check that all filecheck pattern files exist and warn if not found.

    Args:
        cfg: Test configuration dictionary
        test_id: Test identifier for error messages
        yaml_file: YAML filename for error messages
    """

    def check_filechecks(filechecks, where: str) -> None:
        if not filechecks:
            return

        if not isinstance(filechecks, list):
            print(
                f"WARNING: 'filechecks' should be a list in {where}. Found: {type(filechecks).__name__}"
            )
            return

        for pattern_file in filechecks:
            if not isinstance(pattern_file, str):
                print(
                    f"WARNING: filecheck entry should be a string in {where}. Found: {type(pattern_file).__name__}"
                )
                continue

            pattern_path = FILECHECK_DIR / pattern_file
            if not pattern_path.exists():
                print(
                    f"WARNING: filecheck pattern file not found: {pattern_path}"
                    f"\n         Referenced in test '{test_id}' in {yaml_file}"
                )

    # Check top-level filechecks
    if "filechecks" in cfg:
        check_filechecks(cfg["filechecks"], where=f"test '{test_id}' in {yaml_file}")

    # Check filechecks in arch_overrides
    overrides = cfg.get("arch_overrides")
    if isinstance(overrides, dict):
        for arch, arch_cfg in overrides.items():
            if isinstance(arch_cfg, dict) and "filechecks" in arch_cfg:
                check_filechecks(
                    arch_cfg["filechecks"],
                    where=f"arch_overrides['{arch}'] in test '{test_id}' in {yaml_file}",
                )


# Load test configs for a specific framework
def load_framework_test_configs(framework: Framework) -> Dict[str, Any]:
    """Load YAML-based test configurations for a specific framework.

    Args:
        framework: Framework.TORCH or Framework.JAX

    Returns:
    dict: {
        "PLACEHOLDER_MODELS": <dict of placeholder configs>,
        "test_config": <merged dict of all test configs>
    }
    """
    config_dir = Path(__file__).parent / framework.value

    if not config_dir.exists():
        return {"PLACEHOLDER_MODELS": {}, "test_config": {}}

    # Initialize ruamel.yaml loader with duplicate-key checking
    yaml = YAML(typ="rt")
    yaml.allow_duplicate_keys = False

    # Load placeholders only from the dedicated file (only for torch)
    merged_placeholders: Dict[str, Any] = {}
    if framework == Framework.TORCH:
        placeholders_file = config_dir / PLACEHOLDERS_FILENAME
        if placeholders_file.exists():
            with open(placeholders_file, "r") as fh:
                data = yaml.load(fh) or {}
            placeholders = data.get("PLACEHOLDER_MODELS", {}) or {}
            for k, v in placeholders.items():
                entry = dict(v or {})
                merged_placeholders[k] = _normalize_enums_in_cfg(entry)

    # Merge test_config from all YAMLs in the framework directory
    merged_test_config: Dict[str, Any] = {}
    for yaml_file in sorted(config_dir.glob("test_config_*.yaml")):
        if yaml_file.name == PLACEHOLDERS_FILENAME:
            continue
        with open(yaml_file, "r") as fh:
            data = yaml.load(fh) or {}
        tests = data.get("test_config", {}) or {}
        for test_id, cfg in tests.items():
            _validate_allowed_keys(
                cfg, ctx=f"test '{test_id}' in {framework.value}/{yaml_file.name}"
            )
            _validate_filecheck_references(
                cfg,
                test_id=test_id,
                yaml_file=f"{framework.value}/{yaml_file.name}",
            )
            merged_test_config[test_id] = _normalize_enums_in_cfg(cfg)

    return {
        "PLACEHOLDER_MODELS": merged_placeholders,
        "test_config": merged_test_config,
    }


# Load placeholders from dedicated YAML and merge test entries from other YAMLs while validating contents.
def load_all_test_configs() -> Dict[str, Any]:
    """Load and merge YAML-based test configurations for model bring-up and CI.

    This function loads configs from both torch/ and jax/ subdirectories.

    Returns:
    dict: {
        "PLACEHOLDER_MODELS": <dict of placeholder configs>,
        "test_config": <merged dict of all test configs>
    }
    """
    # Load configs from both frameworks
    torch_configs = load_framework_test_configs(Framework.TORCH)
    jax_configs = load_framework_test_configs(Framework.JAX)

    # Merge everything together
    merged_placeholders = (
        torch_configs["PLACEHOLDER_MODELS"] | jax_configs["PLACEHOLDER_MODELS"]
    )
    merged_test_config = torch_configs["test_config"] | jax_configs["test_config"]

    return {
        "PLACEHOLDER_MODELS": merged_placeholders,
        "test_config": merged_test_config,
    }
