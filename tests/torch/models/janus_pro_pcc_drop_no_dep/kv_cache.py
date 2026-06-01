# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""KV cache device alignment (standalone copy of forge ``align_kv_cache_device``)."""

from __future__ import annotations

from typing import Any

import torch


def align_kv_cache_device(past_key_values: Any, device: torch.device | str) -> Any:
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        return past_key_values
    for layer in past_key_values.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        layer.keys = layer.keys.to(device=device)
        layer.values = layer.values.to(device=device)
        layer.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
    return past_key_values
