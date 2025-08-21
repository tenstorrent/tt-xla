# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def pad_inputs(inputs, max_new_tokens):
    batch_size, seq_len = inputs.shape
    max_seq_len = seq_len + max_new_tokens
    padded_inputs = torch.zeros(
        (batch_size, max_seq_len), dtype=inputs.dtype, device=inputs.device
    )
    padded_inputs[:, :seq_len] = inputs
    return padded_inputs, seq_len
