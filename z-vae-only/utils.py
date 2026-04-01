# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal utilities for the VAE decoder standalone directory.

Self-contained -- no imports from sibling directories.
"""

import torch


def calculate_pcc(x, y):
    """Compute Pearson correlation coefficient between two tensors.

    Args:
        x: PyTorch tensor (any shape).
        y: PyTorch tensor (same shape as x).

    Returns:
        Scalar torch.Tensor with PCC value, or NaN if degenerate.
    """
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return torch.tensor(float("nan"))
    return (vx @ vy) / denom
