# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from loguru import logger
import scipy.stats
import torch


def test_pcc():
    org_op = torch.load(
        "/proj_sw/user_dev/kkannan/sep8_mplug_owl2/mPLUG-Owl/mPLUG-Owl2/mplug_owl2/org_f_logits.pt"
    )
    my_op = torch.load(
        "/proj_sw/user_dev/kkannan/aug31_xla_bgd17/tt-xla/my_f_logits.pt"
    )

    logger.info("org_op={}", org_op)
    logger.info("my_op={}", my_op)

    logger.info("allclose={}", torch.allclose(org_op, my_op))

    op1 = org_op.detach().cpu().numpy().reshape(-1)
    op2 = my_op.detach().cpu().numpy().reshape(-1)

    pcc, _ = scipy.stats.pearsonr(op1, op2)
    logger.info("pcc={}", pcc)
