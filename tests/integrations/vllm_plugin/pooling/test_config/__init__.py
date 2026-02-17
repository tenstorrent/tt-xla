# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Load vLLM pooling/embedding model test config from YAML and expose MODEL_CONFIGS
with marks resolved to pytest markers (same pattern as generative test_config).
"""

from pathlib import Path

import pytest

try:
    from ruamel.yaml import YAML
except ImportError:
    YAML = None

# Directory containing this __init__.py
_CONFIG_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS_FILENAME = "model_configs.yaml"

# Defaults for pooling model config. Only "model" is required in YAML; other keys
# are optional overrides. "marks" default to [single_device] when omitted.
# "status" defaults to "unspecified" when omitted.
DEFAULT_POOLING_CONFIG = {
    "task": "embed",
    "dtype": "bfloat16",
    "max_model_len": 128,
    "max_num_batched_tokens": 128,
    "max_num_seqs": 1,
    "disable_sliding_window": True,
    "prompts": ["Hello, my name is"],
    "status": "unspecified",
}

# Map YAML mark names to pytest markers
MARK_MAP = {
    "push": pytest.mark.push,
    "single_device": pytest.mark.single_device,
    "nightly": pytest.mark.nightly,
    "vllm_sweep": pytest.mark.vllm_sweep,
    "tensor_parallel": pytest.mark.tensor_parallel,
    "llmbox": pytest.mark.llmbox,
}


def _load_model_configs_from_file(
    filename: str,
    apply_defaults: bool = False,
    default_config: dict | None = None,
    default_marks: list | None = None,
) -> dict:
    """Load model_configs from a specific YAML file and resolve mark names to pytest marks.

    When apply_defaults is True, configs are merged with default_config (or
    DEFAULT_POOLING_CONFIG if not provided) and marks default to default_marks
    (or [single_device] if not provided). When apply_defaults is False, no merge
    and marks are required in YAML.
    """
    if YAML is None:
        raise ImportError("ruamel.yaml is required to load vLLM pooling test config")
    yaml_path = _CONFIG_DIR / filename
    if not yaml_path.exists():
        return {}
    yaml = YAML(typ="safe")
    with open(yaml_path, "r") as f:
        data = yaml.load(f) or {}
    raw = data.get("model_configs", data)
    if not isinstance(raw, dict):
        return {}
    result = {}
    base_config = (
        default_config if default_config is not None else DEFAULT_POOLING_CONFIG
    )
    marks_if_default = default_marks if default_marks is not None else ["single_device"]
    use_default_marks = marks_if_default if apply_defaults else []
    for name, cfg in raw.items():
        cfg = dict(cfg or {})
        mark_names = cfg.pop("marks", use_default_marks)
        marks = [MARK_MAP[m] for m in mark_names if m in MARK_MAP]
        if apply_defaults:
            merged = {**base_config, **cfg}
            merged["marks"] = marks
            result[name] = merged
        else:
            cfg["marks"] = marks
            cfg.setdefault("status", "unspecified")
            result[name] = cfg
    return result


def _load_model_configs() -> dict:
    """Load model_configs from YAML and resolve mark names to pytest marks."""
    return _load_model_configs_from_file(MODEL_CONFIGS_FILENAME, apply_defaults=True)


MODEL_CONFIGS = _load_model_configs()


def get_model_config_params():
    """Return list of pytest.param(model_name, config, id=name, marks=...) for parametrize."""
    return [
        pytest.param(name, cfg, id=name, marks=cfg.get("marks", []))
        for name, cfg in MODEL_CONFIGS.items()
    ]
