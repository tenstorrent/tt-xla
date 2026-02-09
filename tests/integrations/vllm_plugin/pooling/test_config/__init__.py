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

_CONFIG_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS_FILENAME = "model_configs.yaml"

# Map YAML mark names to pytest markers
MARK_MAP = {
    "push": pytest.mark.push,
    "single_device": pytest.mark.single_device,
    "nightly": pytest.mark.nightly,
    "vllm_sweep": pytest.mark.vllm_sweep,
    "tensor_parallel": pytest.mark.tensor_parallel,
    "llmbox": pytest.mark.llmbox,
}


def _load_model_configs() -> dict:
    """Load model_configs from YAML and resolve mark names to pytest marks."""
    if YAML is None:
        return {}
    yaml_path = _CONFIG_DIR / MODEL_CONFIGS_FILENAME
    if not yaml_path.exists():
        return {}
    yaml = YAML(typ="safe")
    with open(yaml_path, "r") as f:
        data = yaml.load(f) or {}
    raw = data.get("model_configs", data)
    if not isinstance(raw, dict):
        return {}
    result = {}
    for name, cfg in raw.items():
        cfg = dict(cfg or {})
        mark_names = cfg.pop("marks", [])
        marks = [MARK_MAP[m] for m in mark_names if m in MARK_MAP]
        cfg["marks"] = marks
        result[name] = cfg
    return result


MODEL_CONFIGS = _load_model_configs()


def get_model_config_params():
    """Return list of pytest.param(model_name, config, id=name, marks=...) for parametrize."""
    return [
        pytest.param(name, cfg, id=name, marks=cfg.get("marks", []))
        for name, cfg in MODEL_CONFIGS.items()
    ]
