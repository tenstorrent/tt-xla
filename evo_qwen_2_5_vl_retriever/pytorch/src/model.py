# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
EvoQwen2.5-VL-Retriever model wrapper for extracting embeddings from model outputs.
"""

import torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        return outputs
