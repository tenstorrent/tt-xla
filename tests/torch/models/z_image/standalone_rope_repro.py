# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal Z-Image RoPE reproducers with synthetic inputs — no checkpoint / transformer load.

Requires tt-mlir branch ``akannan/zimage_shlo_bug`` for gather/index on complex tables.

Matches ``ZImageTransformer2DModel`` defaults (``transformer_z_image.py``):
  axes_dims=[32, 48, 48], axes_lens=[1024, 512, 512], rope_theta=256.0

Token counts follow the single-chip slice bisect (issue #4756):
  image path pos_ids length 3616 → mlir ``512x24`` / ``concatenate.65`` class failures.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Z-Image transformer RoPE config (diffusers defaults for Tongyi-MAI/Z-Image).
ROPE_THETA = 256.0
AXES_DIMS = (32, 48, 48)
AXES_LENS = (1024, 512, 512)

# Synthetic sequence lengths (same order of magnitude as patchify+pad capture).
SYNTH_IMAGE_NUM_TOKENS = 3616
SYNTH_CAP_NUM_TOKENS = 512

def make_synthetic_pos_ids(
    num_tokens: int,
    *,
    seed: int = 42,
    axes_lens: tuple[int, ...] = AXES_LENS,
) -> torch.Tensor:
    """``[N, 3]`` int32 position ids with valid indices per axis."""
    gen = torch.Generator().manual_seed(seed)
    cols = [
        torch.randint(0, axes_lens[i], (num_tokens,), generator=gen, dtype=torch.int32)
        for i in range(len(axes_lens))
    ]
    return torch.stack(cols, dim=-1)


def precompute_freqs_cis(
    axes_dims: tuple[int, ...] = AXES_DIMS,
    axes_lens: tuple[int, ...] = AXES_LENS,
    theta: float = ROPE_THETA,
) -> list[torch.Tensor]:
    """Same tables as ``RopeEmbedder.precompute_freqs_cis`` (CPU, complex64)."""
    freqs_cis = []
    for d, e in zip(axes_dims, axes_lens):
        freqs = 1.0 / (
            theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
        )
        timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
    return freqs_cis


class StandaloneRopeEmbedderModule(nn.Module):
    """``RopeEmbedder.__call__``: runtime precompute + per-axis index + cat."""

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
        self._freqs_cis: list[torch.Tensor] | None = None

    def _ensure_tables(self, device: torch.device) -> list[torch.Tensor]:
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis(
                tuple(self.axes_dims), tuple(self.axes_lens), self.theta
            )
            self._freqs_cis = [t.to(device) for t in self._freqs_cis]
        elif self._freqs_cis[0].device != device:
            self._freqs_cis = [t.to(device) for t in self._freqs_cis]
        return self._freqs_cis

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        assert pos_ids.ndim == 2 and pos_ids.shape[-1] == len(self.axes_dims)
        tables = self._ensure_tables(pos_ids.device)
        parts = [tables[i][pos_ids[:, i].long()] for i in range(len(self.axes_dims))]
        return torch.cat(parts, dim=-1)


class StandaloneRopeIndexAxisModule(nn.Module):
    """``freqs_cis[axis][pos_ids[:, axis]]`` with lifted precomputed tables."""

    def __init__(self, axis_idx: int):
        super().__init__()
        tables = precompute_freqs_cis()
        self.axis_idx = axis_idx
        self.register_buffer("freqs_cis_table", tables[axis_idx])

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        return self.freqs_cis_table[pos_ids[:, self.axis_idx].long()]


class StandaloneRopeIndexAndCatModule(nn.Module):
    """Index all axes from precomputed tables, then ``torch.cat`` (no runtime polar)."""

    def __init__(self):
        super().__init__()
        tables = precompute_freqs_cis()
        for i, table in enumerate(tables):
            self.register_buffer(f"freqs_cis_{i}", table)

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        parts = [
            getattr(self, f"freqs_cis_{i}")[pos_ids[:, i].long()]
            for i in range(len(AXES_DIMS))
        ]
        return torch.cat(parts, dim=-1)


class StandaloneRopePolarThenIndexAxis1Module(nn.Module):
    """``torch.polar`` for axis 1, then gather — reproduces ``loc('p1.5')`` class."""

    def __init__(self, axis_idx: int = 1):
        super().__init__()
        self.theta = ROPE_THETA
        self.d = AXES_DIMS[axis_idx]
        self.e = AXES_LENS[axis_idx]
        self.axis_idx = axis_idx

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        device = pos_ids.device
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.d, 2, dtype=torch.float64, device=device) / self.d)
        )
        timestep = torch.arange(self.e, device=device, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        table = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
        return table[pos_ids[:, self.axis_idx].long()]


class StandaloneRopePolarAxisModule(nn.Module):
    """Single-axis ``torch.polar`` table only (negative control — should compile on TT)."""

    def __init__(self, axis_idx: int = 1):
        super().__init__()
        self.theta = ROPE_THETA
        self.d = AXES_DIMS[axis_idx]
        self.e = AXES_LENS[axis_idx]

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        device = _.device
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.d, 2, dtype=torch.float64, device=device) / self.d)
        )
        timestep = torch.arange(self.e, device=device, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)


def build_standalone_case(name: str) -> tuple[nn.Module, list[torch.Tensor]]:
    """Return ``(module, inputs)`` for a named standalone repro case."""
    image_pos = make_synthetic_pos_ids(SYNTH_IMAGE_NUM_TOKENS)
    cap_pos = make_synthetic_pos_ids(SYNTH_CAP_NUM_TOKENS, seed=43)
    dummy = torch.zeros(1, dtype=torch.float32)

    if name == "rope_embedder_image":
        return StandaloneRopeEmbedderModule(), [image_pos]
    if name == "rope_embedder_cap":
        return StandaloneRopeEmbedderModule(), [cap_pos]
    if name == "rope_index_axis0":
        return StandaloneRopeIndexAxisModule(axis_idx=0), [image_pos]
    if name == "rope_index_axis1":
        return StandaloneRopeIndexAxisModule(axis_idx=1), [image_pos]
    if name == "rope_index_axis2":
        return StandaloneRopeIndexAxisModule(axis_idx=2), [image_pos]
    if name == "rope_index_and_cat":
        return StandaloneRopeIndexAndCatModule(), [image_pos]
    if name == "rope_polar_then_index_axis1":
        return StandaloneRopePolarThenIndexAxis1Module(axis_idx=1), [image_pos]
    if name == "rope_polar_axis1":
        return StandaloneRopePolarAxisModule(axis_idx=1), [dummy]
    raise ValueError(f"unknown standalone case: {name}")


RUN_ON_TT_CASES = frozenset(
    {
        "rope_embedder_image",
        "rope_embedder_cap",
        "rope_index_axis0",
        "rope_index_axis1",
        "rope_index_axis2",
        "rope_index_and_cat",
        "rope_polar_then_index_axis1",
        "rope_polar_axis1",
    }
)

ALL_STANDALONE_CASES = tuple(sorted(RUN_ON_TT_CASES))
