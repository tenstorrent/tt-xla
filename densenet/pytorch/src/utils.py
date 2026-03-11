# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


def op_norm(outputs, op_threshs):
    """Normalize outputs according to operating points for a given model.

    XLA-compatible version: uses torch.where instead of boolean mask indexing
    to avoid dynamo graph breaks that cause 'fused_N has no attribute xla_args'.

    Args:
        outputs: outputs of self.classifier(). torch.Size(batch_size, num_tasks)
        op_threshs: torch.Size(num_tasks) or broadcastable thresholds.
    Returns:
        outputs_new: normalized outputs, torch.Size(batch_size, num_tasks)
    """
    op_threshs = op_threshs.expand(outputs.shape[0], -1)

    not_nan = ~torch.isnan(op_threshs)

    scaled_leq = outputs / (op_threshs * 2)
    scaled_gt = 1.0 - ((1.0 - outputs) / ((1.0 - op_threshs) * 2))

    is_leq = outputs < op_threshs
    result = torch.where(is_leq, scaled_leq, scaled_gt)

    default = torch.full_like(outputs, 0.5)
    return torch.where(not_nan, result, default)
