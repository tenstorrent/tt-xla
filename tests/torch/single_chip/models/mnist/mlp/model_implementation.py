# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTMLPModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 classes for MNIST

    def forward(self, x):
        # Flatten the input
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x
