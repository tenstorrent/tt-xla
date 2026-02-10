# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Load vLLM generative model test config from YAML and expose MODEL_CONFIGS
with marks resolved to pytest markers (similar to tests/runner/test_config).
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

# Defaults for generative model config. Only "model" is required in YAML; other keys
# are optional overrides. "marks" default to [single_device] when omitted.
# "status" defaults to "unspecified" when omitted.
DEFAULT_GENERATIVE_CONFIG = {
    "max_num_batched_tokens": 128,
    "max_num_seqs": 1,
    "max_model_len": 128,
    "gpu_memory_utilization": 0.002,
    "trust_remote_code": True,
    "additional_config": {
        "enable_const_eval": False,
        "min_context_len": 32,
    },
    "prompts": ["Hello, my name is"],
    "status": "unspecified",
}

# Tensor parallel llmbox: same shape as DEFAULT_GENERATIVE_CONFIG but TP-specific values.
# Used when loading model_configs_tensor_parallel_llmbox.yaml so the YAML can list only overrides.
DEFAULT_TENSOR_PARALLEL_LLMBOX_CONFIG = {
    "max_num_batched_tokens": 32,
    "max_num_seqs": 1,
    "max_model_len": 32,
    "gpu_memory_utilization": 0.002,
    "trust_remote_code": True,
    "additional_config": {
        "enable_const_eval": False,
        "min_context_len": 32,
        "enable_tensor_parallel": True,
    },
    "prompts": ["Hello, my name is"],
    "status": "unspecified",
}

# Map YAML mark names to pytest markers
MARK_MAP = {
    "push": pytest.mark.push,
    "nightly": pytest.mark.nightly,
    "vllm_sweep": pytest.mark.vllm_sweep,
}


def _load_model_configs_from_file(
    filename: str,
    apply_defaults: bool = False,
    default_config: dict | None = None,
    default_marks: list | None = None,
) -> dict:
    """Load model_configs from a specific YAML file and resolve mark names to pytest marks.

    When apply_defaults is True, configs are merged with default_config (or
    DEFAULT_GENERATIVE_CONFIG if not provided) and marks default to default_marks
    (or [single_device] if not provided). When apply_defaults is False, no merge
    and marks are required in YAML.
    """
    if YAML is None:
        raise ImportError("ruamel.yaml is required to load vLLM generative test config")
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
        default_config if default_config is not None else DEFAULT_GENERATIVE_CONFIG
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


# Loaded config: model name -> config dict (with "marks" as list of pytest marks)
MODEL_CONFIGS = _load_model_configs()


def get_model_config_params():
    """Return list of pytest.param(model_name, config, id=name, marks=...) for parametrize."""
    return [
        pytest.param(name, cfg, id=name, marks=cfg.get("marks", []))
        for name, cfg in MODEL_CONFIGS.items()
    ]


# Tensor parallel llmbox configs (defaults applied like single-device; YAML has only overrides)
MODEL_CONFIGS_TP_LLMBOX_FILENAME = "model_configs_tensor_parallel_llmbox.yaml"
MODEL_CONFIGS_TP_LLMBOX = _load_model_configs_from_file(
    MODEL_CONFIGS_TP_LLMBOX_FILENAME,
    apply_defaults=True,
    default_config=DEFAULT_TENSOR_PARALLEL_LLMBOX_CONFIG,
    default_marks=["tensor_parallel"],
)


def get_model_config_params_tp_llmbox():
    """Return list of pytest.param for tensor parallel llmbox models."""
    return [
        pytest.param(name, cfg, id=name, marks=cfg.get("marks", []))
        for name, cfg in MODEL_CONFIGS_TP_LLMBOX.items()
    ]
