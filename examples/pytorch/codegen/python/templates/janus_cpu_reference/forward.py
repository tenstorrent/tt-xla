# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU golden entry point — delegates to tt-xla ``Layer0LnAttnNoDep``."""

from __future__ import annotations

import torch

from cpu_reference.ttxla_golden import run_layer0_ln_attn_no_dep_stacked

VARIANT = "Pro_1B"
DTYPE = torch.bfloat16


def run_forward_from_fixtures(variant: str = VARIANT) -> torch.Tensor:
    """Stacked stages ``[3, batch, seq, hidden]`` — same tensor as codegen CPU path."""
    stacked = run_layer0_ln_attn_no_dep_stacked(variant)
    if stacked.ndim != 4 or stacked.shape[0] != 3:
        raise ValueError(f"Expected [3, B, S, H], got {tuple(stacked.shape)}")
    return stacked.to(dtype=DTYPE)
