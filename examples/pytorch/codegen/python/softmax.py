# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.runtime as xr
from tt_torch import codegen_py

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")


class SoftmaxModule(torch.nn.Module):
    def forward(self, attention_scores):
        return F.softmax(attention_scores, dim=-1)


# Load attention scores (input to softmax)
attention_scores = torch.load("/home/tt-xla/attention_scores_cpu.pt")

op = SoftmaxModule()

codegen_py(op, attention_scores, export_path="softmaxmodule")
