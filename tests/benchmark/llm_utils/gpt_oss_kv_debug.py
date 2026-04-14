# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Optional KV cache stats for GPT-OSS Galaxy debugging.

Set ``GPT_OSS_LAYER_DIAGNOSE_DEBUG`` to ``1``, ``true``, ``yes``, or ``on`` to print
layer-0 ``StaticCache`` key/value min/max/mean abs and ``all_zero``. Used by
``scripts/gpt_oss_galaxy_layer_diagnose.py`` and
``scripts/bisect_gpt_oss_layer_cpu_attn_tt_moe_decode.py`` (CPU KV only).
"""

from __future__ import annotations

import os


def layer_diagnose_debug_enabled() -> bool:
    v = os.environ.get("GPT_OSS_LAYER_DIAGNOSE_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def debug_kv_report(past_key_values, label: str, *, flush_xla: bool) -> None:
    """Print layer-0 cache tensor stats. Use ``flush_xla=True`` when tensors live on XLA."""

    if not layer_diagnose_debug_enabled():
        return

    import torch

    if flush_xla:
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    lay0 = past_key_values.layers[0]
    for name, t in (("keys", lay0.keys), ("values", lay0.values)):
        tc = t.detach().float().cpu()
        print(
            f"[debug-kv] {label} layer0 {name}: shape={tuple(tc.shape)} "
            f"min={float(tc.min().item()):.6g} max={float(tc.max().item()):.6g} "
            f"mean_abs={float(tc.abs().mean().item()):.6g} "
            f"all_zero={bool((tc == 0).all().item())}"
        )
