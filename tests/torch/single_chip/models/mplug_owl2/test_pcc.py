# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from loguru import logger
import scipy.stats
import torch


def test_pcc():
    my_op_new = torch.load(
        "/proj_sw/user_dev/kkannan/aug31_xla_bgd17/tt-xla/my_new_logits.pt"
    )
    my_op = torch.load(
        "/proj_sw/user_dev/kkannan/aug31_xla_bgd17/tt-xla/my_f_logits.pt"
    )

    logger.info("my_op_new={}", my_op_new)
    logger.info("my_op={}", my_op)

    logger.info("allclose={}", torch.allclose(my_op_new, my_op))

    op1 = my_op_new.detach().cpu().numpy().reshape(-1)
    op2 = my_op.detach().cpu().numpy().reshape(-1)

    pcc, _ = scipy.stats.pearsonr(op1, op2)
    logger.info("pcc={}", pcc)
