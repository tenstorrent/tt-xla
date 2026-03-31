# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


def calculate_pcc(x, y):
    """Calculate Pearson Correlation Coefficient between two tensors."""
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    x_flat, y_flat = x.flatten().float(), y.flatten().float()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom
