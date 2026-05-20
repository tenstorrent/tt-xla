# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Configuration for streaming inference runs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

Mode = Literal["whole_graph", "layer_eager"]


@dataclass
class StreamingConfig:
    """Run-time knobs for `streaming.core.run_streaming`.

    All fields have working defaults; env-var overrides are picked up via
    `StreamingConfig.from_env()`.
    """

    # ---- inference shape ----
    prompt_len: int = 128
    max_new_tokens: int = 3
    batch_size: int = 8
    num_layers: Optional[int] = 43

    # ---- mode ----
    # whole_graph: streaming load → one `torch.compile(model)` → prefill + decode
    # layer_eager: streaming load → per-layer compile + sequential execute
    mode: Mode = "whole_graph"

    # ---- weight dtype overrides ----
    # Adapter interprets these values (e.g. "bfp_bf8", "bfp_bf4") into
    # model-specific {path: dtype} via `adapter.weight_dtype_overrides(...)`.
    # "bf16" / "" / "none" → no override for that weight class.
    expert_dtype: str = "bf16"  # MoE expert weights
    attn_dtype: str = "bf16"  # Attention weights

    @classmethod
    def from_env(cls) -> "StreamingConfig":
        mode_val = os.environ.get("STREAM_MODE", "whole_graph").strip().lower()
        if mode_val not in ("whole_graph", "layer_eager"):
            raise ValueError(
                f"STREAM_MODE must be 'whole_graph' or 'layer_eager', "
                f"got {mode_val!r}"
            )
        num_layers_env = os.environ.get("STREAM_NUM_LAYERS", "43")
        return cls(
            prompt_len=int(os.environ.get("STREAM_PROMPT_LEN", "128")),
            max_new_tokens=int(os.environ.get("STREAM_MAX_NEW_TOKENS", "3")),
            batch_size=int(os.environ.get("STREAM_BATCH_SIZE", "8")),
            num_layers=int(num_layers_env) if num_layers_env else None,
            mode=mode_val,  # type: ignore[arg-type]
            expert_dtype=os.environ.get("STREAM_EXPERT_DTYPE", "bf16").strip(),
            attn_dtype=os.environ.get("STREAM_ATTN_DTYPE", "bf16").strip(),
        )
