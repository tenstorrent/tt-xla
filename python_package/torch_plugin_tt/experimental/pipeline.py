# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Explicit layer-split pipeline parallelism across TT devices.

The user declares the split by passing one ``nn.Module`` per stage; stage ``i``
runs on ``torch_xla.device(i)``. The activation crosses each boundary with
``h.to(next_device)``, which the tt-xla plugin performs as a device-to-device
socket transfer (TT-Metal MeshSocket) when both endpoints are TT submeshes of
the same parent mesh.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_xla


class PipelineParallel(nn.Module):
    """N stages, one per TT device. Forward threads the activation across
    devices: x -> xla:0 [stage0] -> xla:1 [stage1] -> ... -> cpu."""

    def __init__(self, stages: list[nn.Module]):
        super().__init__()
        if len(stages) < 2:
            raise ValueError("pipeline needs at least 2 stages")
        self.devices = [torch_xla.device(i) for i in range(len(stages))]
        self.stages = nn.ModuleList(
            stage.to(self.devices[i]) for i, stage in enumerate(stages)
        )

    def stage_param_devices(self) -> list[str]:
        """Device each stage's weights permanently live on (proves placement)."""
        return [str(next(s.parameters()).device) for s in self.stages]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.to(self.devices[0])
        for i, stage in enumerate(self.stages):
            h = stage(h)
            if i + 1 < len(self.stages):
                h = h.to(self.devices[i + 1])  # cross-device activation transfer
        return h.to("cpu")


def pipeline(stages: list[nn.Module]) -> PipelineParallel:
    """Factory: build a PipelineParallel from a list of per-stage modules."""
    return PipelineParallel(stages)
