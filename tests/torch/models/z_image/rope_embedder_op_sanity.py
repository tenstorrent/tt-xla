# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-op sanities for ``RopeEmbedder.precompute_freqs_cis`` + index + cat (issue #4756).

Each module mirrors one step from ``transformer_z_image.py`` (axis 1 by default
for steps 1–4; full 3-axis config for 5–8). Uses synthetic ``pos_ids`` — no model load.

Steps:
  1. ``1.0 / (theta ** (arange(0, d, 2) / d))``
  2. ``torch.arange(e)``
  3. ``torch.outer(timestep, freqs).float()``
  4. ``torch.polar(...).to(complex64)``
  5. whole ``precompute_freqs_cis`` (all axes)
  6. ``table[pos_ids[:, i]]`` (one axis, precomputed table)
  7. ``for i in range(len(axes_dims)): gather`` (no cat; returns axis-1 slice)
  8. ``torch.cat`` on the three gathered tensors (precomputed tables)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .standalone_rope_repro import (
    AXES_DIMS,
    AXES_LENS,
    ROPE_THETA,
    SYNTH_IMAGE_NUM_TOKENS,
    make_synthetic_pos_ids,
    precompute_freqs_cis,
)

# Default bisect axis: matches ``512x24`` table in mlir logs (axes_dims[1]=48, axes_lens[1]=512).
DEFAULT_AXIS_IDX = 1

def _axis_params(axis_idx: int) -> tuple[float, int, int]:
    return ROPE_THETA, AXES_DIMS[axis_idx], AXES_LENS[axis_idx]


def make_polar_freqs_cis_table_for_axis(
    axis_idx: int = DEFAULT_AXIS_IDX,
    *,
    theta: float = ROPE_THETA,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Exact per-axis RoPE table from ``RopeEmbedder.precompute_freqs_cis`` step 4:

    ``freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)``

    Axis 1 shape is ``[512, 24]`` (= ``[axes_lens[1], axes_dims[1] // 2]``).
    """
    _theta, d, e = _axis_params(axis_idx)
    freqs = 1.0 / (
        theta ** (torch.arange(0, d, 2, dtype=torch.float64, device=device) / d)
    )
    timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
    freqs = torch.outer(timestep, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)


class RopeOp01InvFreqModule(nn.Module):
    """Step 1 only: inverse-frequency vector length ``d/2``."""

    def __init__(self, axis_idx: int = DEFAULT_AXIS_IDX):
        super().__init__()
        self.theta, self.d, _e = _axis_params(axis_idx)

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        device = _.device
        return 1.0 / (
            self.theta
            ** (torch.arange(0, self.d, 2, dtype=torch.float64, device=device) / self.d)
        )


class RopeOp02TimestepModule(nn.Module):
    """Step 2 only: ``torch.arange(e)``."""

    def __init__(self, axis_idx: int = DEFAULT_AXIS_IDX):
        super().__init__()
        _theta, _d, self.e = _axis_params(axis_idx)

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        return torch.arange(self.e, device=_.device, dtype=torch.float64)


class RopeOp03OuterModule(nn.Module):
    """Steps 1–3: ends at ``torch.outer(timestep, freqs).float()`` (no polar)."""

    def __init__(self, axis_idx: int = DEFAULT_AXIS_IDX):
        super().__init__()
        self.theta, self.d, self.e = _axis_params(axis_idx)

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        device = _.device
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.d, 2, dtype=torch.float64, device=device) / self.d)
        )
        timestep = torch.arange(self.e, device=device, dtype=torch.float64)
        return torch.outer(timestep, freqs).float()


class RopeOp04PolarModule(nn.Module):
    """Steps 1–4: ends at ``torch.polar(...).to(complex64)`` for one axis table."""

    def __init__(self, axis_idx: int = DEFAULT_AXIS_IDX):
        super().__init__()
        self.theta, self.d, self.e = _axis_params(axis_idx)

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        device = _.device
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.d, 2, dtype=torch.float64, device=device) / self.d)
        )
        timestep = torch.arange(self.e, device=device, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)


class RopeOp05PrecomputeAllAxesModule(nn.Module):
    """Step 5: full ``precompute_freqs_cis`` for all axes (polar only, no index/cat)."""

    def __init__(
        self,
        theta: float = ROPE_THETA,
        axes_dims: tuple[int, ...] = AXES_DIMS,
        axes_lens: tuple[int, ...] = AXES_LENS,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        device = _.device
        tables: list[torch.Tensor] = []
        for d, e in zip(self.axes_dims, self.axes_lens):
            freqs = 1.0 / (
                self.theta
                ** (torch.arange(0, d, 2, dtype=torch.float64, device=device) / d)
            )
            timestep = torch.arange(e, device=device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            tables.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return torch.cat([t.reshape(-1) for t in tables])


class RopeGatherComplexPolarTableOnlyModule(nn.Module):
    """
    Confirmation repro: **gather on complex64 polar table only**.

    - ``polar_table`` is the exact ``torch.polar`` output (lifted buffer, not computed in forward).
    - Forward is a single gather: ``polar_table[pos_ids[:, axis]]``.
    - No ``torch.polar`` / arange / outer / cat in the compiled graph.

    If this fails TT compile, the issue is **gather (index) on complex tensors**, not polar.
    """

    def __init__(self, axis_idx: int = DEFAULT_AXIS_IDX):
        super().__init__()
        table = make_polar_freqs_cis_table_for_axis(axis_idx)
        e, half_d = AXES_LENS[axis_idx], AXES_DIMS[axis_idx] // 2
        if tuple(table.shape) != (e, half_d):
            raise ValueError(f"expected polar table [{e}, {half_d}], got {tuple(table.shape)}")
        self.axis_idx = axis_idx
        self.register_buffer("polar_table", table)

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        return self.polar_table[pos_ids[:, self.axis_idx].long()]


class RopeOp06GatherOneAxisModule(RopeGatherComplexPolarTableOnlyModule):
    """Step 6 — same as :class:`RopeGatherComplexPolarTableOnlyModule`."""


class RopeOp07GatherLoopNoCatModule(nn.Module):
    """Step 7: embedder for-loop gathers all axes; no ``torch.cat`` (output = axis 1 part)."""

    def __init__(self):
        super().__init__()
        tables = precompute_freqs_cis()
        for i, table in enumerate(tables):
            self.register_buffer(f"freqs_cis_{i}", table)
        self.n_axes = len(AXES_DIMS)
        self.return_axis = DEFAULT_AXIS_IDX

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i in range(self.n_axes):
            index = pos_ids[:, i].long()
            parts.append(getattr(self, f"freqs_cis_{i}")[index])
        return parts[self.return_axis]


class RopeOp08CatGatheredModule(nn.Module):
    """Step 8: for-loop gather each axis, then ``torch.cat(..., dim=-1)``."""

    def __init__(self):
        super().__init__()
        tables = precompute_freqs_cis()
        for i, table in enumerate(tables):
            self.register_buffer(f"freqs_cis_{i}", table)
        self.n_axes = len(AXES_DIMS)

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i in range(self.n_axes):
            index = pos_ids[:, i].long()
            parts.append(getattr(self, f"freqs_cis_{i}")[index])
        return torch.cat(parts, dim=-1)


def build_op_sanity_case(name: str) -> tuple[nn.Module, list[torch.Tensor]]:
    """``(module, inputs)`` for ``run_op_test``."""
    pos_ids = make_synthetic_pos_ids(SYNTH_IMAGE_NUM_TOKENS)
    dummy = torch.zeros(1, dtype=torch.float32)

    builders: dict[str, tuple[nn.Module, list[torch.Tensor]]] = {
        "op01_inv_freq": (RopeOp01InvFreqModule(), [dummy]),
        "op02_timestep": (RopeOp02TimestepModule(), [dummy]),
        "op03_outer": (RopeOp03OuterModule(), [dummy]),
        "op04_polar": (RopeOp04PolarModule(), [dummy]),
        "op05_precompute_all_axes": (RopeOp05PrecomputeAllAxesModule(), [dummy]),
        "op06_gather_axis1": (RopeOp06GatherOneAxisModule(DEFAULT_AXIS_IDX), [pos_ids]),
        "gather_complex_polar_table_only": (
            RopeGatherComplexPolarTableOnlyModule(DEFAULT_AXIS_IDX),
            [pos_ids],
        ),
        "op07_gather_loop_no_cat": (RopeOp07GatherLoopNoCatModule(), [pos_ids]),
        "op08_cat_gathered": (RopeOp08CatGatheredModule(), [pos_ids]),
    }
    if name not in builders:
        raise ValueError(f"unknown op sanity case: {name}")
    return builders[name]


# All steps: pass on TT when tt-mlir is on branch akannan/zimage_shlo_bug.
RUN_ON_TT_CASES = frozenset(
    {
        "op01_inv_freq",
        "op02_timestep",
        "op03_outer",
        "op04_polar",
        "op05_precompute_all_axes",
        "op06_gather_axis1",
        "gather_complex_polar_table_only",
        "op07_gather_loop_no_cat",
        "op08_cat_gathered",
    }
)

ALL_OP_SANITY_CASES = tuple(sorted(RUN_ON_TT_CASES))
