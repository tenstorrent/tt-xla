# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


def op_norm(outputs, op_threshs):
    """Normalize outputs according to operating points for a given model.
    Args:
        outputs: outputs of self.classifier(). torch.Size(batch_size, num_tasks)
        op_threshs_arr: torch.Size(batch_size, num_tasks) with self.op_threshs expanded.
    Returns:
        outputs_new: normalized outputs, torch.Size(batch_size, num_tasks)
    """
    # expand to batch size so we can do parallel comp
    op_threshs = op_threshs.expand(outputs.shape[0], -1)

    # initial values will be 0.5
    outputs_new = torch.zeros(outputs.shape, device=outputs.device) + 0.5

    # only select non-nan elements otherwise the gradient breaks
    mask_leq = (outputs < op_threshs) & ~torch.isnan(op_threshs)
    mask_gt = ~(outputs < op_threshs) & ~torch.isnan(op_threshs)

    # scale outputs less than thresh
    outputs_new[mask_leq] = outputs[mask_leq] / (op_threshs[mask_leq] * 2)
    # scale outputs greater than thresh
    outputs_new[mask_gt] = 1.0 - (
        (1.0 - outputs[mask_gt]) / ((1 - op_threshs[mask_gt]) * 2)
    )

    return outputs_new
