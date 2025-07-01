# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNNNoDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5)

        self.conv2 = nn.Conv2d(32, 64, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5)

        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.bn3 = nn.BatchNorm1d(256, eps=1e-5)

        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128, eps=1e-5)

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x
