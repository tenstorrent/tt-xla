# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Load vLLM pooling/embedding model test config from YAML (same pattern as generative test_config).
Config is populated by scripts/add_group1_to_vllm_configs.py from group1 passing tests
when task is NLP_EMBED_GEN.
"""

from pathlib import Path

try:
    from ruamel.yaml import YAML
except ImportError:
    YAML = None

_CONFIG_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS_FILENAME = "model_configs.yaml"


def _load_model_configs() -> dict:
    """Load model_configs from YAML."""
    if YAML is None:
        return {}
    yaml_path = _CONFIG_DIR / MODEL_CONFIGS_FILENAME
    if not yaml_path.exists():
        return {}
    yaml = YAML(typ="safe")
    with open(yaml_path, "r") as f:
        data = yaml.load(f) or {}
    return data.get("model_configs", data) or {}


MODEL_CONFIGS = _load_model_configs()
