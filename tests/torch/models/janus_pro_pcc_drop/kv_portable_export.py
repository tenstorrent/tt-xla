# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Portable fixture sidecars for tt-metal ``cpu_reference`` (tensor-only KV + JSON config)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

PORTABLE_KV_FILENAME = "past_key_values_portable.pt"
CONFIG_JSON_FILENAME = "llama_config.json"


def extract_kv_tensors(
    past_key_values: Any,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    if past_key_values is None:
        return [], [], 0
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        keys = [k.detach().cpu().clone() for k in past_key_values.key_cache]
        values = [v.detach().cpu().clone() for v in past_key_values.value_cache]
        seen = int(getattr(past_key_values, "_seen_tokens", 0))
        if hasattr(past_key_values, "get_seq_length"):
            try:
                seen = int(past_key_values.get_seq_length())
            except TypeError:
                pass
        return keys, values, seen
    if hasattr(past_key_values, "layers"):
        keys, values = [], []
        for layer in past_key_values.layers:
            if getattr(layer, "is_initialized", False):
                keys.append(layer.keys.detach().cpu().clone())
                values.append(layer.values.detach().cpu().clone())
            else:
                keys.append(torch.tensor([]))
                values.append(torch.tensor([]))
        return keys, values, int(past_key_values.get_seq_length())
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def save_past_key_values_portable(past_key_values: Any, path: Path) -> None:
    keys, values, seen = extract_kv_tensors(past_key_values)
    torch.save(
        {"key_cache": keys, "value_cache": values, "seen_tokens": seen},
        path,
    )


def save_llama_config_json(config: Any, path: Path) -> None:
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n")


def save_fixture_sidecars(
    output_dir: Path,
    past_key_values: Any,
    config: Any,
) -> None:
    save_past_key_values_portable(
        past_key_values, output_dir / PORTABLE_KV_FILENAME
    )
    save_llama_config_json(config, output_dir / CONFIG_JSON_FILENAME)


def _populate_dynamic_cache_layers(
    cache: Any,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
) -> Any:
    """Fill a transformers 5.x ``DynamicCache`` (``.layers`` API, no ``key_cache``)."""
    from transformers.cache_utils import DynamicLayer

    for key_states, value_states in zip(keys, values):
        layer = DynamicLayer()
        if key_states.numel() > 0:
            layer.dtype = key_states.dtype
            layer.device = key_states.device
            layer.keys = key_states
            layer.values = value_states
            layer.is_initialized = True
        cache.layers.append(layer)
    return cache


def load_past_key_values_portable(path: Path) -> Any:
    """Rebuild ``DynamicCache`` for the installed transformers (tensor-only sidecar)."""
    from transformers.cache_utils import DynamicCache

    data = torch.load(path, weights_only=False)
    keys: list[torch.Tensor] = data["key_cache"]
    values: list[torch.Tensor] = data["value_cache"]
    seen = int(data.get("seen_tokens", 0))

    probe = DynamicCache()
    if hasattr(probe, "key_cache"):
        cache = DynamicCache()
        for key_states, value_states in zip(keys, values):
            cache.key_cache.append(key_states)
            cache.value_cache.append(value_states)
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = seen
        return cache

    if hasattr(probe, "layers"):
        cache = DynamicCache()
        return _populate_dynamic_cache_layers(cache, keys, values)

    raise TypeError(
        f"Unsupported DynamicCache layout: {type(probe)} "
        f"(no key_cache or layers)"
    )
